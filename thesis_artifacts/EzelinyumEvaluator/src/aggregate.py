"""Aggregate per-model reports into a cross-model comparison.

Outputs:
- outputs/comparison/headline.csv
- outputs/comparison/per_topic.csv
- outputs/comparison/per_city.csv
- outputs/comparison/per_difficulty.csv
- outputs/comparison/report.md   (narrative summary)
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from registry import REGISTRY, all_keys  # noqa


OUT_ROOT = Path("/home/ain480/EzelinyumEvaluator/outputs")
COMPARISON = OUT_ROOT / "comparison"
COMPARISON.mkdir(parents=True, exist_ok=True)


def load_reports() -> dict[str, dict]:
    reports = {}
    for key in all_keys():
        p = OUT_ROOT / key / "report.json"
        if p.exists():
            with open(p) as f:
                reports[key] = json.load(f)
    return reports


def write_headline(reports: dict[str, dict]):
    fp = COMPARISON / "headline.csv"
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_key", "display", "family", "n", "accuracy", "ci95_low", "ci95_high",
            "n_hedged", "n_refused", "n_unparseable", "n_missing_images",
        ])
        for key, r in reports.items():
            o = r["overall"]
            m = r["model"]
            w.writerow([
                key, m["display"], m["family"], o["n"],
                round(o["accuracy"], 4), round(o["ci95_low"], 4), round(o["ci95_high"], 4),
                o.get("n_hedged", 0), o.get("n_refused", 0),
                o.get("n_unparseable", 0), o.get("n_missing_images", 0),
            ])
    print(f"wrote {fp}")


def write_per_topic(reports: dict[str, dict]):
    fp = COMPARISON / "per_topic.csv"
    topics = sorted({t for r in reports.values() for t in r["per_topic"].keys()})
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key"] + [f"{t}__acc" for t in topics] + [f"{t}__n" for t in topics])
        for key, r in reports.items():
            row = [key]
            for t in topics:
                m = r["per_topic"].get(t, {})
                row.append(round(m.get("accuracy", 0.0), 4) if m else "")
            for t in topics:
                m = r["per_topic"].get(t, {})
                row.append(m.get("n", "") if m else "")
            w.writerow(row)
    print(f"wrote {fp}")


def write_per_city(reports: dict[str, dict]):
    fp = COMPARISON / "per_city.csv"
    cities = sorted({c for r in reports.values() for c in r["per_city"].keys()})
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key"] + cities)
        for key, r in reports.items():
            row = [key]
            for c in cities:
                m = r["per_city"].get(c, {})
                row.append(round(m.get("accuracy", 0.0), 4) if m else "")
            w.writerow(row)
    print(f"wrote {fp}")


def write_per_difficulty(reports: dict[str, dict]):
    fp = COMPARISON / "per_difficulty.csv"
    diffs = sorted({d for r in reports.values() for d in r["per_difficulty"].keys()})
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key"] + diffs)
        for key, r in reports.items():
            row = [key]
            for d in diffs:
                m = r["per_difficulty"].get(d, {})
                row.append(round(m.get("accuracy", 0.0), 4) if m else "")
            w.writerow(row)
    print(f"wrote {fp}")


def write_per_city_type(reports: dict[str, dict]):
    fp = COMPARISON / "per_city_type.csv"
    cts = sorted({c for r in reports.values() for c in r["per_city_type"].keys()})
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key"] + cts)
        for key, r in reports.items():
            row = [key]
            for c in cts:
                m = r["per_city_type"].get(c, {})
                row.append(round(m.get("accuracy", 0.0), 4) if m else "")
            w.writerow(row)
    print(f"wrote {fp}")


def paired_mcnemar(preds_a: list[bool], preds_b: list[bool]) -> dict:
    """McNemar's test for paired binary outcomes (same sample order)."""
    import math
    b = sum(1 for a, c in zip(preds_a, preds_b) if a and not c)
    c = sum(1 for a, d in zip(preds_a, preds_b) if not a and d)
    # Continuity-corrected.
    n_disc = b + c
    if n_disc == 0:
        return dict(b=b, c=c, chi2=0.0, p=1.0)
    chi2 = ((abs(b - c) - 1) ** 2) / n_disc
    # p-value: 1 - chi2 cdf with df=1.
    # Use erf-based approximation: P(chi2 >= x) = erfc(sqrt(x/2)).
    p = math.erfc(math.sqrt(chi2 / 2))
    return dict(b=b, c=c, chi2=chi2, p=p)


def load_predictions(key: str) -> list[dict]:
    fp = OUT_ROOT / key / "predictions.jsonl"
    if not fp.exists():
        return []
    rows = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_pairwise_mcnemar(reports: dict[str, dict]):
    """Compute pairwise McNemar between all evaluated models."""
    keys = sorted(reports.keys())
    preds: dict[str, dict[str, bool]] = {}
    for k in keys:
        preds[k] = {r["question_id"]: r["correct"] for r in load_predictions(k)}
    # Common question_ids across all models.
    common = set.intersection(*[set(p.keys()) for p in preds.values()]) if preds else set()
    sorted_q = sorted(common)

    fp = COMPARISON / "pairwise_mcnemar.csv"
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "n", "b_a_only_right", "c_b_only_right", "chi2", "p"])
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pa = [preds[a][q] for q in sorted_q]
                pb = [preds[b][q] for q in sorted_q]
                m = paired_mcnemar(pa, pb)
                w.writerow([a, b, len(sorted_q), m["b"], m["c"], round(m["chi2"], 3), f"{m['p']:.4g}"])
    print(f"wrote {fp}")


def write_narrative(reports: dict[str, dict]):
    fp = COMPARISON / "report.md"
    # Sort models by accuracy desc.
    items = sorted(reports.items(), key=lambda kv: kv[1]["overall"]["accuracy"], reverse=True)

    lines = []
    lines.append("# EzelinyumEvaluator — Cross-model Comparison")
    lines.append("")
    lines.append(f"Models evaluated: **{len(items)}**. Benchmark: 5,240 held-out questions across 40 cities, 14 topics.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Rank | Model | Family | Acc | 95% CI | Hedged | Refused |")
    lines.append("|---|---|---|---:|---|---:|---:|")
    for rank, (key, r) in enumerate(items, 1):
        o = r["overall"]
        m = r["model"]
        lines.append(
            f"| {rank} | **{m['display']}** | {m['family']} | "
            f"{o['accuracy']:.3f} | [{o['ci95_low']:.3f}, {o['ci95_high']:.3f}] | "
            f"{o.get('n_hedged', 0)} | {o.get('n_refused', 0)} |"
        )
    lines.append("")

    # Per-topic ranking on the topics where ablations differ.
    lines.append("## Per-topic accuracy")
    lines.append("")
    topics = sorted({t for r in reports.values() for t in r["per_topic"].keys()})
    header = "| Topic | Rand | Maj | " + " | ".join(items[i][0] for i in range(min(len(items), 15))) + " |"
    sep = "|---|" + "---|" * (2 + min(len(items), 15))
    lines.append(header)
    lines.append(sep)
    for t in topics:
        first = items[0][1]["per_topic"].get(t, {})
        rand = first.get("random_baseline")
        maj = first.get("majority_baseline")
        rand_s = f"{rand:.2f}" if rand is not None else "—"
        maj_s = f"{maj:.2f}" if maj is not None else "—"
        cells = []
        for k, _ in items[:15]:
            r = reports[k]
            m = r["per_topic"].get(t, {})
            cells.append(f"{m.get('accuracy', 0):.2f}" if m else "—")
        lines.append(f"| {t} | {rand_s} | {maj_s} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per city-type (seen vs unseen).
    lines.append("## Seen vs Unseen cities")
    lines.append("")
    lines.append("| Model | Seen acc | Unseen acc | Gap |")
    lines.append("|---|---:|---:|---:|")
    for key, r in items:
        seen = r["per_city_type"].get("seen", {}).get("accuracy", 0)
        unseen = r["per_city_type"].get("unseen", {}).get("accuracy", 0)
        lines.append(f"| {r['model']['display']} | {seen:.3f} | {unseen:.3f} | {seen - unseen:+.3f} |")
    lines.append("")

    # Per difficulty.
    lines.append("## Per-difficulty")
    lines.append("")
    diffs = ["easy", "medium", "hard"]
    lines.append("| Model | " + " | ".join(diffs) + " |")
    lines.append("|---|" + "---:|" * len(diffs))
    for key, r in items:
        cells = []
        for d in diffs:
            cells.append(f"{r['per_difficulty'].get(d, {}).get('accuracy', 0):.3f}")
        lines.append(f"| {r['model']['display']} | " + " | ".join(cells) + " |")

    fp.write_text("\n".join(lines))
    print(f"wrote {fp}")


def main():
    reports = load_reports()
    if not reports:
        print("No reports found yet.")
        return
    print(f"Loaded reports for: {sorted(reports.keys())}")
    write_headline(reports)
    write_per_topic(reports)
    write_per_city(reports)
    write_per_difficulty(reports)
    write_per_city_type(reports)
    write_pairwise_mcnemar(reports)
    write_narrative(reports)


if __name__ == "__main__":
    main()
