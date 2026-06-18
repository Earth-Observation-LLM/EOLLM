#!/usr/bin/env python3
"""Build the cross-model benchmark-modes report — MAXIMAL detail.

Reads every results/<model>/<model>_summary.json plus the per-(model,mode)
report JSONs produced by run_modes_vllm.py, and emits, under results/:

  modes_report.md     human-readable tables (the deliverable)
  modes_report.json   the same numbers, machine-readable
  by_task.csv         tidy long form: model,mode,topic,accuracy,correct,total
  overall.csv         tidy long form: model,mode,n,accuracy,ci_low,ci_high

Tables in the .md:
  1. Overall accuracy   — model x mode  (with bootstrap 90% CI on 'full')
  2. Modality ablation  — per model: full / sat_only / sv_only / blind + deltas
                          (full - blind = vision gain; full - sat_only / full -
                          sv_only = does the dropped view matter?)
  3. Per-task accuracy   — for EACH mode: model x topic matrix
  4. Per-difficulty      — model x mode x {easy,medium,hard}
  5. Per-city-type       — model x mode x {seen,unseen} (generalization)

Everything is read from the on-disk reports; this script does no GPU work and
can run locally after pull_results.sh.
"""
import argparse
import csv
import json
from pathlib import Path

MODE_ORDER = ["full", "sat_only", "sv_only", "blind"]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def discover(results_dir):
    """Return {model_name: {"summary": {...}, "reports": {mode: report}}}."""
    out = {}
    for d in sorted(Path(results_dir).iterdir()):
        if not d.is_dir():
            continue
        model = d.name
        summary_p = d / f"{model}_summary.json"
        if not summary_p.exists():
            continue
        summary = load_json(summary_p)
        reports = {}
        for mode in summary.get("modes", {}):
            rp = d / f"{model}_{mode}_report.json"
            if rp.exists():
                reports[mode] = load_json(rp)
        out[model] = {"summary": summary, "reports": reports}
    return out


def modes_present(data):
    """Ordered union of modes across all models."""
    seen = []
    for m in data.values():
        for mode in m["summary"].get("modes", {}):
            if mode not in seen:
                seen.append(mode)
    return [mode for mode in MODE_ORDER if mode in seen] + [m for m in seen if m not in MODE_ORDER]


def all_topics(data):
    topics = set()
    for m in data.values():
        for rep in m["reports"].values():
            topics.update(rep.get("by_topic", {}).keys())
    return sorted(topics)


def fmt_pct(x):
    return f"{100 * x:.1f}" if x is not None else "—"


def acc_of(report, *, key=None, bucket=None):
    """Pull an accuracy float out of a report. key in {None(overall),
    'by_topic','by_difficulty','by_city_type'}; bucket selects the sub-key."""
    if report is None:
        return None
    if key is None:
        return report.get("overall", {}).get("accuracy")
    sub = report.get(key, {}).get(bucket)
    return sub.get("accuracy") if sub else None


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build(results_dir):
    data = discover(results_dir)
    if not data:
        raise SystemExit(f"no per-model summaries found under {results_dir}/<model>/")
    models = list(data.keys())
    modes = modes_present(data)
    topics = all_topics(data)

    md = ["# Benchmark-modes sweep — cross-model report",
          "",
          f"Models: {', '.join(models)}",
          f"Modes: {', '.join(modes)}",
          "Thinking: **OFF** (letter-only output) for every model.",
          ""]

    jreport = {"models": models, "modes": modes, "topics": topics,
               "overall": {}, "by_topic": {}, "by_difficulty": {},
               "by_city_type": {}, "ablation_delta": {}, "meta": {}}

    for model in models:
        s = data[model]["summary"]
        jreport["meta"][model] = {
            "model_id": s.get("model_id"), "thinking": s.get("thinking"),
            "max_model_len": s.get("max_model_len"), "max_num_seqs": s.get("max_num_seqs"),
        }

    # ---- Table 1: overall accuracy, model x mode (+ CI on full) -------------
    md += ["## 1. Overall accuracy (model × mode)", ""]
    headers = ["model"] + modes + ["full 90% CI", "n(full)"]
    rows = []
    for model in models:
        reps = data[model]["reports"]
        cells = [model]
        for mode in modes:
            cells.append(fmt_pct(acc_of(reps.get(mode))))
            jreport["overall"].setdefault(model, {})[mode] = acc_of(reps.get(mode))
        full = reps.get("full")
        ci = full.get("confidence_interval") if full else None
        ci_s = f"{fmt_pct(ci['p05'])}–{fmt_pct(ci['p95'])}" if ci else "—"
        n_full = full.get("overall", {}).get("total") if full else None
        cells += [ci_s, n_full if n_full is not None else "—"]
        rows.append(cells)
    md += [md_table(headers, rows), ""]

    # ---- Table 2: modality ablation deltas per model ------------------------
    md += ["## 2. Modality ablation (per model)",
           "",
           "`full−blind` = total vision gain. `full−sat_only` / `full−sv_only` = "
           "how much accuracy is lost when the *other* view is removed (≈0 means "
           "that view was redundant). Percentage-point differences.",
           ""]
    headers = ["model", "full", "sat_only", "sv_only", "blind",
               "full−blind", "full−sat_only", "full−sv_only"]
    rows = []
    for model in models:
        reps = data[model]["reports"]
        a = {m: acc_of(reps.get(m)) for m in ["full", "sat_only", "sv_only", "blind"]}

        def delta(x, y):
            if a.get(x) is None or a.get(y) is None:
                return "—"
            return f"{100 * (a[x] - a[y]):+.1f}"
        rows.append([model, fmt_pct(a["full"]), fmt_pct(a["sat_only"]),
                     fmt_pct(a["sv_only"]), fmt_pct(a["blind"]),
                     delta("full", "blind"), delta("full", "sat_only"),
                     delta("full", "sv_only")])
        jreport["ablation_delta"][model] = a
    md += [md_table(headers, rows), ""]

    # ---- Table 3: per-task accuracy, one matrix per mode --------------------
    md += ["## 3. Per-task accuracy (model × topic), one table per mode", ""]
    for mode in modes:
        # only topics that appear in this mode (sat_only/sv_only cover a subset)
        mode_topics = sorted({t for model in models
                              for t in data[model]["reports"].get(mode, {}).get("by_topic", {})})
        if not mode_topics:
            continue
        md += [f"### mode = `{mode}`", ""]
        headers = ["topic"] + models
        rows = []
        for t in mode_topics:
            cells = [t]
            for model in models:
                rep = data[model]["reports"].get(mode)
                cells.append(fmt_pct(acc_of(rep, key="by_topic", bucket=t)))
                jreport["by_topic"].setdefault(mode, {}).setdefault(t, {})[model] = \
                    acc_of(rep, key="by_topic", bucket=t)
            rows.append(cells)
        md += [md_table(headers, rows), ""]

    # ---- Table 4: per-difficulty, model x mode x difficulty -----------------
    md += ["## 4. Per-difficulty accuracy (model × mode)", ""]
    diffs = ["easy", "medium", "hard"]
    headers = ["model", "mode"] + diffs
    rows = []
    for model in models:
        for mode in modes:
            rep = data[model]["reports"].get(mode)
            if rep is None:
                continue
            cells = [model, mode]
            for dl in diffs:
                cells.append(fmt_pct(acc_of(rep, key="by_difficulty", bucket=dl)))
                jreport["by_difficulty"].setdefault(model, {}).setdefault(mode, {})[dl] = \
                    acc_of(rep, key="by_difficulty", bucket=dl)
            rows.append(cells)
    md += [md_table(headers, rows), ""]

    # ---- Table 5: per-city-type (seen vs unseen) ----------------------------
    md += ["## 5. Generalization: seen vs unseen cities (model × mode)",
           "",
           "`benchmark_city_type` split. Gap = seen − unseen (pp).", ""]
    headers = ["model", "mode", "seen", "unseen", "seen−unseen"]
    rows = []
    for model in models:
        for mode in modes:
            rep = data[model]["reports"].get(mode)
            if rep is None:
                continue
            seen = acc_of(rep, key="by_benchmark_city_type", bucket="seen")
            unseen = acc_of(rep, key="by_benchmark_city_type", bucket="unseen")
            gap = f"{100 * (seen - unseen):+.1f}" if (seen is not None and unseen is not None) else "—"
            rows.append([model, mode, fmt_pct(seen), fmt_pct(unseen), gap])
            jreport["by_city_type"].setdefault(model, {})[mode] = {"seen": seen, "unseen": unseen}
    md += [md_table(headers, rows), ""]

    # ---- write outputs ------------------------------------------------------
    rd = Path(results_dir)
    (rd / "modes_report.md").write_text("\n".join(md))
    with open(rd / "modes_report.json", "w") as f:
        json.dump(jreport, f, indent=2)

    # tidy CSVs
    with open(rd / "by_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "topic", "accuracy", "correct", "total"])
        for model in models:
            for mode, rep in data[model]["reports"].items():
                for t, a in rep.get("by_topic", {}).items():
                    w.writerow([model, mode, t, a["accuracy"], a["correct"], a["total"]])
    with open(rd / "overall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "n", "accuracy", "ci_p05", "ci_p95"])
        for model in models:
            for mode, rep in data[model]["reports"].items():
                ov = rep.get("overall", {})
                ci = rep.get("confidence_interval", {})
                w.writerow([model, mode, ov.get("total"), ov.get("accuracy"),
                            ci.get("p05"), ci.get("p95")])

    print(f"wrote {rd}/modes_report.md (+ .json, by_task.csv, overall.csv)")
    print(f"  models={len(models)} modes={len(modes)} topics={len(topics)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="dir with <model>/ subdirs")
    args = ap.parse_args()
    build(args.results)
