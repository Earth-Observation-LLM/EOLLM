"""
Merge multiple pipeline run outputs into a single unified dataset.

Usage:
    python src/merge_datasets.py [--output OUTPUT_PATH]

Discovers all dataset.jsonl files from known run directories, deduplicates
by sample_id (latest run wins on conflict), and writes a merged JSONL file.
"""

import json
import os
import sys
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All known run directories, in chronological order (oldest first).
# Later entries win on duplicate sample_id conflicts.
RUN_DIRS = [
    "/home/alperen/Documents/EODATA/output_1445/output",
    "/home/alperen/Documents/EODATA/output_673img/output",
    "/home/alperen/Documents/EODATA/output_707/output",
    "/home/alperen/Documents/EODATA/output_1459_28_03/output",
]

# Add new run directories here as needed.


def load_run(jsonl_path: str, run_label: str) -> dict:
    """Load a dataset.jsonl into a dict keyed by sample_id."""
    records = {}
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {run_label} line {lineno}: JSON parse error — {e}")
                skipped += 1
                continue
            sid = record.get("sample_id")
            if not sid:
                print(f"  [WARN] {run_label} line {lineno}: missing sample_id, skipping")
                skipped += 1
                continue
            records[sid] = record
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description="Merge EOLLM pipeline run outputs")
    parser.add_argument(
        "--output",
        default="/home/alperen/Documents/EODATA/dataset_merged.jsonl",
        help="Output path for the merged dataset (default: /home/alperen/Documents/EODATA/dataset_merged.jsonl)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-run stats"
    )
    args = parser.parse_args()

    merged: dict = {}          # sample_id -> record (latest run wins)
    run_stats = []             # per-run summary
    total_skipped = 0

    print("=" * 60)
    print("EOLLM Dataset Merge")
    print("=" * 60)

    for run_dir in RUN_DIRS:
        jsonl_path = os.path.join(run_dir, "dataset.jsonl")
        label = os.path.basename(os.path.dirname(run_dir)) if run_dir.endswith("/output") else os.path.basename(run_dir)

        if not os.path.isfile(jsonl_path):
            print(f"\n[SKIP] {label}: no dataset.jsonl found at {jsonl_path}")
            continue

        print(f"\n[LOAD] {label}")
        records, skipped = load_run(jsonl_path, label)
        total_skipped += skipped

        # Track which were new vs overwritten
        new_count = 0
        overwritten = 0
        cities = defaultdict(int)
        for sid, rec in records.items():
            if sid in merged:
                overwritten += 1
            else:
                new_count += 1
            merged[sid] = rec
            city = rec.get("location", {}).get("city", "Unknown")
            cities[city] += 1

        stat = {
            "label": label,
            "path": jsonl_path,
            "total": len(records),
            "new": new_count,
            "overwritten": overwritten,
            "skipped": skipped,
            "cities": dict(cities),
        }
        run_stats.append(stat)

        if not args.quiet:
            print(f"  Loaded   : {len(records):,} samples")
            print(f"  New      : {new_count:,}")
            print(f"  Overwrote: {overwritten:,} (duplicate sample_ids)")
            print(f"  Cities   : {', '.join(sorted(cities))}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total unique samples : {len(merged):,}")
    print(f"Total skipped lines  : {total_skipped:,}")

    # City breakdown
    all_cities = defaultdict(int)
    topic_counts = defaultdict(int)
    question_total = 0
    for rec in merged.values():
        city = rec.get("location", {}).get("city", "Unknown")
        all_cities[city] += 1
        # Count questions in the 'questions' array
        qs = rec.get("questions", [])
        question_total += len(qs)
        for q in qs:
            topic_counts[q.get("topic", "unknown")] += 1

    print(f"\nTotal questions (all_questions): {question_total:,}")
    print(f"Avg questions per sample       : {question_total / len(merged):.1f}")

    print(f"\nCity breakdown ({len(all_cities)} cities):")
    for city, count in sorted(all_cities.items(), key=lambda x: -x[1]):
        print(f"  {city:<30} {count:>5}")

    print(f"\nQuestion type distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / question_total if question_total else 0
        print(f"  {topic:<25} {count:>7,}  ({pct:.1f}%)")

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nWriting merged dataset to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in merged.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done. {len(merged):,} samples written ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
