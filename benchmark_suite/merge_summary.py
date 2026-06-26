#!/usr/bin/env python3
"""merge_summary.py — rebuild combined views after a per-mode run.

When run.py is invoked once per mode (to bound host RAM on a large benchmark),
each process writes correct per-mode files (<mode>_predictions.jsonl,
<mode>_report.json) but overwrites the dir-level summary.json / all_predictions
with only its own mode. This reassembles those from whatever per-mode reports and
predictions are present, so the output dir looks identical to a single 4-mode run.

    python merge_summary.py results/<key>
"""
import json
import sys
from pathlib import Path

MODES = ["full", "sat_only", "sv_only", "blind"]


def main(out_dir):
    out_dir = Path(out_dir)
    meta = json.load(open(out_dir / "meta.json")) if (out_dir / "meta.json").exists() else {}
    summary = {"model": meta.get("key"), "display": meta.get("display"),
               "backend": meta.get("backend"), "modes": {}}
    all_rows = []
    for mode in MODES:
        rep_p = out_dir / f"{mode}_report.json"
        pred_p = out_dir / f"{mode}_predictions.jsonl"
        if not rep_p.exists():
            continue
        report = json.load(open(rep_p))
        n = 0
        with open(pred_p) as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))
                    n += 1
        summary["modes"][mode] = {
            "n": n, "overall": report["overall"],
            "by_topic": {t: a["accuracy"] for t, a in report.get("by_topic", {}).items()},
        }
        print(f"  [{mode}] n={n} acc={report['overall']['accuracy']:.4f}")

    with open(out_dir / "all_predictions.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"merged {len(summary['modes'])} modes, {len(all_rows)} rows -> {out_dir}/summary.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qwen35_9b_local")
