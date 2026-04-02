#!/usr/bin/env python3
"""
Dataset Splitting Pipeline for EOLLM.

Transforms a hierarchical dataset_merged.jsonl (one record per location with
nested questions) into four flat JSONL files for train/val/benchmark splits.

Usage:
    python splitting/split_dataset.py --input output/dataset.jsonl --outdir splits/
    python splitting/split_dataset.py --input /path/to/dataset_merged.jsonl
"""

import argparse
import json
import os
import sys

# Allow imports from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from splitting.flatten import flatten_dataset
from splitting.filters import (
    filter_low_streetview,
    remove_question_types,
    split_mismatch_subtypes,
    deduplicate_per_location,
    filter_mismatch_leaks,
)
from splitting.splitter import extract_benchmark, split_seen_unseen, split_per_city
from splitting.downsampler import stratified_downsample
from splitting.stats_report import generate_report

# ─── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

UNSEEN_CITIES = {
    "Istanbul",
    "Moscow",
    "Chicago",
    "Seoul",
    "Rio de Janeiro",
    "Cape Town",
    "Sydney",
    "Singapore",
}

SEEN_CITIES = {
    "Toronto", "Paris", "Budapest", "Barcelona", "Athens",
    "Brussels", "Helsinki", "Vienna", "Tallinn", "Amsterdam",
    "Reykjavik", "New York City", "Tokyo", "Buenos Aires",
    "Mexico City", "Lisbon", "Stockholm", "Zurich",
    "St. Petersburg", "Berlin", "London", "Rome", "Taipei",
    "Nicosia", "Antalya", "Bursa", "Samsun", "Ankara",
    "Kayseri", "Mumbai", "Izmir", "Nairobi",
}

QUESTION_TARGETS = {
    "camera_direction"      : 2000,
    "mismatch_binary_easy"  : 2000,
    "mismatch_binary_hard"  : 2000,
    "mismatch_mcq_easy"     : 2000,
    "mismatch_mcq_hard"     : 2000,
    "road_type"             : 2000,
    "land_use"              : 2000,
    "urban_density"         : 2000,
    "amenity_richness"      : 2000,
    "transit_density"       : 2000,
    "junction_type"         : 2000,
    "road_surface"          : 2000,
    "green_space"           : 2000,
    "building_height"       : None,  # Keep ALL
}

REMOVE_QUESTION_TYPES = ["water_proximity"]

BENCHMARK_RATIO = 0.10
MIN_SV_ANGLES = 4


# ─── I/O Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Split EOLLM dataset into train/val/benchmark")
    parser.add_argument("--input", required=True, help="Path to dataset_merged.jsonl")
    parser.add_argument("--outdir", default="splits", help="Output directory (default: splits/)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--strategy",
        choices=["seen_unseen", "per_city"],
        default="seen_unseen",
        help="Split strategy: 'seen_unseen' (unseen cities -> val) or 'per_city' (N%% per city -> val)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio per city (only for --strategy per_city, default: 0.15)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    records = load_jsonl(args.input)
    total_questions = sum(len(r.get("questions", [])) for r in records)
    print(f"  Loaded {len(records)} locations, {total_questions} total questions")

    # Print unique cities
    cities_found = sorted(set(r.get("location", {}).get("city", "?") for r in records))
    print(f"  Cities found ({len(cities_found)}): {', '.join(cities_found)}")

    # ── Step 2: Filter low streetview ─────────────────────────────────────
    print(f"\n[STEP 2] Filtering locations with streetview_count < {MIN_SV_ANGLES}...")
    records, _, _ = filter_low_streetview(records, MIN_SV_ANGLES)

    # ── Step 3: Flatten ───────────────────────────────────────────────────
    print("\n[STEP 3] Flattening to question-level records...")
    flat = flatten_dataset(records)

    # ── Step 4: Remove excluded question types ────────────────────────────
    print(f"\n[STEP 4] Removing question types: {REMOVE_QUESTION_TYPES}...")
    flat, _ = remove_question_types(flat, REMOVE_QUESTION_TYPES)

    # ── Step 5: Mismatch subtype assignment ───────────────────────────────
    print("\n[STEP 5] Splitting mismatch subtypes (easy/hard)...")
    flat = split_mismatch_subtypes(flat)

    # ── Step 5b: Deduplicate multi-variant topics ─────────────────────────
    print("\n[STEP 5b] Deduplicating to 1 question per topic per location...")
    flat, _ = deduplicate_per_location(flat, args.seed)

    # Check targets vs actual counts
    for topic, target in QUESTION_TARGETS.items():
        if target is None:
            continue
        actual = sum(1 for r in flat if r["topic"] == topic)
        if actual < target:
            print(f"  WARNING: {topic} has {actual} questions, target is {target}. Adjusting target to {actual}.")
            QUESTION_TARGETS[topic] = actual

    # ── Step 6: Assign city types ─────────────────────────────────────────
    print("\n[STEP 6] Assigning city types (seen/unseen)...")
    seen_count = 0
    unseen_count = 0
    for rec in flat:
        if rec["city"] in UNSEEN_CITIES:
            rec["city_type"] = "unseen"
            unseen_count += 1
        else:
            rec["city_type"] = "seen"
            seen_count += 1
    print(f"  Seen city questions: {seen_count}")
    print(f"  Unseen city questions: {unseen_count}")

    # ── Step 7a: Benchmark split (SAME for all strategies) ─────────────
    print("\n[STEP 7a] Benchmark split (10% per city, location-level)...")
    benchmark, remaining = extract_benchmark(
        flat, UNSEEN_CITIES, BENCHMARK_RATIO, args.seed
    )

    # ── Step 7b: Train/val split (strategy-dependent) ────────────────────
    if args.strategy == "seen_unseen":
        print("\n[STEP 7b] Train/val split: seen/unseen city strategy...")
        remaining_val, remaining_train = split_seen_unseen(
            remaining, UNSEEN_CITIES, SEEN_CITIES
        )

        # ── Step 8: Leak filter on train (only for seen_unseen) ───────────
        print("\n[STEP 8] Mismatch easy leak filter (train only)...")
        remaining_train, binary_disc, mcq_disc = filter_mismatch_leaks(
            remaining_train, UNSEEN_CITIES
        )
        leak_stats = {"binary_discarded": binary_disc, "mcq_discarded": mcq_disc}

        # Re-check targets after leak filter
        for topic in ["mismatch_binary_easy", "mismatch_mcq_easy"]:
            actual = sum(1 for r in remaining_train if r["topic"] == topic)
            target = QUESTION_TARGETS.get(topic)
            if target is not None and actual < target:
                print(f"  WARNING: {topic} has {actual} after leak filter, adjusting target from {target}")
                QUESTION_TARGETS[topic] = actual

    elif args.strategy == "per_city":
        print(f"\n[STEP 7b] Train/val split: per-city sampling (val_ratio={args.val_ratio})...")
        remaining_val, remaining_train = split_per_city(
            remaining, val_ratio=args.val_ratio, seed=args.seed
        )
        leak_stats = {"binary_discarded": 0, "mcq_discarded": 0}
        print("\n[STEP 8] Skipping leak filter (not applicable for per_city strategy)")

    # ── Step 9: Stratified downsampling (train only) ──────────────────────
    print("\n[STEP 9] Stratified downsampling (train only)...")
    train = stratified_downsample(remaining_train, QUESTION_TARGETS, args.seed)
    validation = remaining_val

    # ── Step 10: Add metadata fields ──────────────────────────────────────
    print("\n[STEP 10] Adding metadata fields to all records...")
    for rec in train:
        rec["split"] = "train"
        rec.setdefault("city_type", "unseen" if rec["city"] in UNSEEN_CITIES else "seen")
        rec.setdefault("benchmark_city_type", None)

    for rec in validation:
        rec["split"] = "validation"
        rec.setdefault("city_type", "unseen" if rec["city"] in UNSEEN_CITIES else "seen")
        rec.setdefault("benchmark_city_type", None)

    for rec in benchmark:
        rec["split"] = "benchmark"
        rec.setdefault("city_type", "unseen" if rec["city"] in UNSEEN_CITIES else "seen")
        # benchmark_city_type already set by splitter

    # ── Step 11: Export ───────────────────────────────────────────────────
    print("\n[STEP 11] Exporting files...")
    save_jsonl(train, os.path.join(args.outdir, "train.jsonl"))
    save_jsonl(validation, os.path.join(args.outdir, "validation.jsonl"))
    save_jsonl(benchmark, os.path.join(args.outdir, "benchmark_with_answers.jsonl"))

    # benchmark_public: null out answer
    benchmark_public = []
    for rec in benchmark:
        pub = dict(rec)
        pub["answer"] = None
        benchmark_public.append(pub)
    save_jsonl(benchmark_public, os.path.join(args.outdir, "benchmark_public.jsonl"))

    # ── Step 12: Statistics report ────────────────────────────────────────
    print("\n[STEP 12] Generating statistics report...")
    report_path = os.path.join(args.outdir, "split_statistics_report.txt")
    report = generate_report(
        train, validation, benchmark,
        UNSEEN_CITIES, SEEN_CITIES,
        leak_stats, report_path,
        strategy=args.strategy,
    )
    print(f"\n{report}")

    print(f"\nDone! Output files in {args.outdir}/")


if __name__ == "__main__":
    main()
