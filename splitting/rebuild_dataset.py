#!/usr/bin/env python3
"""
Rebuild EODATA dataset with split-first, generate-second methodology.

Splits LOCATIONS first, then generates questions per split so that mismatch
distractors can only reference locations within the same split.  This prevents
cross-split image leaks by construction.

Pipeline:
  1. Load base samples from dataset_merged.jsonl (strip existing questions)
  2. Filter locations with < 4 streetview angles
  3. Split LOCATIONS: benchmark (10% per city) → remaining
  4. For each strategy (seen_unseen, per_city):
       Split remaining → train locations + val locations
  5. For each split: generate composites (Step 07) + questions (Step 05)
  6. Flatten → filter → deduplicate → downsample (train) → export
  7. Package benchmark self-contained + strategy dirs with images

Usage:
    python splitting/rebuild_dataset.py \\
        --input /home/alperen/Documents/EODATA/dataset_merged.jsonl \\
        --outdir /mnt/hdd/EODATA_final \\
        --image-dirs /home/alperen/Documents/EODATA/output_1445/output \\
                     /home/alperen/Documents/EODATA/output_1459_28_03/output \\
                     /home/alperen/Documents/EODATA/output_673img/output \\
                     /home/alperen/Documents/EODATA/output_707/output
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

# Allow imports from project root and src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from splitting.flatten import flatten_record
from splitting.filters import (
    remove_question_types,
    split_mismatch_subtypes,
    deduplicate_per_location,
    filter_mismatch_leaks,
)
from splitting.downsampler import stratified_downsample
from splitting.stats_report import generate_report
from splitting.package_splits import collect_image_paths, find_source

# ─── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

UNSEEN_CITIES = {
    "Istanbul", "Moscow", "Chicago", "Seoul",
    "Rio de Janeiro", "Cape Town", "Sydney", "Singapore",
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

REMOVE_TOPICS = ["water_proximity"]
BENCHMARK_RATIO = 0.10
MIN_SV_ANGLES = 4


# ─── I/O Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


# ─── Base sample extraction ──────────────────────────────────────────────────

def load_base_samples(jsonl_path):
    """Load location records, stripping pre-generated questions and variants."""
    records = load_jsonl(jsonl_path)
    base = []
    for rec in records:
        sample = dict(rec)
        # Remove pre-generated questions and variant metadata
        sample.pop("questions", None)
        sample.pop("all_questions", None)
        sample.pop("mismatch_binary_variants", None)
        sample.pop("mismatch_mcq_variants", None)
        sample.pop("camera_arrow_paths", None)
        sample.pop("sat_marked_path", None)
        sample.pop("stv_composite_path", None)
        sample.pop("stv_composite_labeled_path", None)
        # Keep: sample_id, location, images, metadata, validation

        # Promote metadata fields to top level so step05/07 can read them.
        # build_jsonl_record() nests OSM fields under "metadata", but
        # generate_questions() and generate_composites() read from the top level.
        meta = sample.get("metadata", {})
        for key in [
            "land_use_category", "osm_building_count", "osm_has_park",
            "osm_amenity_count", "osm_amenity_types", "osm_road_surface",
            "osm_junction_type", "osm_water_distance_m", "osm_transit_stop_count",
            "road_bearing",
        ]:
            if key in meta:
                sample.setdefault(key, meta[key])
        # Key renames: JSONL uses different names than step05 expects
        if "road_type" in meta:
            sample.setdefault("highway_type", meta["road_type"])
        if "osm_median_building_levels" in meta:
            sample.setdefault("osm_median_levels", meta["osm_median_building_levels"])

        base.append(sample)
    return base


def get_city(sample):
    """Extract canonical city name from sample."""
    return sample.get("location", {}).get("city", sample.get("city", ""))


def get_sv_count(sample):
    """Get streetview count from sample."""
    return sample.get("validation", {}).get("streetview_count", 0)


# ─── Location-level splitting ────────────────────────────────────────────────

def split_locations_benchmark(samples, ratio=0.10, seed=42):
    """Split locations into benchmark + remaining (10% per city).

    Returns (benchmark_sids: set, remaining_sids: set)
    """
    rng = random.Random(seed)
    city_to_sids = defaultdict(list)
    for s in samples:
        city_to_sids[get_city(s)].append(s["sample_id"])

    benchmark_sids = set()
    remaining_sids = set()

    print(f"\n  {'City':<25s} {'Total':>6s} {'Bench':>6s} {'Remain':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")

    for city in sorted(city_to_sids.keys()):
        sids = sorted(set(city_to_sids[city]))
        rng.shuffle(sids)
        n_bench = max(1, round(len(sids) * ratio))
        benchmark_sids.update(sids[:n_bench])
        remaining_sids.update(sids[n_bench:])
        print(f"  {city:<25s} {len(sids):>6d} {n_bench:>6d} {len(sids)-n_bench:>6d}")

    print(f"\n  Benchmark locations: {len(benchmark_sids)}")
    print(f"  Remaining locations: {len(remaining_sids)}")
    return benchmark_sids, remaining_sids


def split_locations_seen_unseen(remaining_sids, sid_to_city):
    """Split remaining into train (seen cities) and val (unseen cities)."""
    train_sids = set()
    val_sids = set()
    for sid in remaining_sids:
        city = sid_to_city[sid]
        if city in UNSEEN_CITIES:
            val_sids.add(sid)
        else:
            train_sids.add(sid)
    print(f"  seen_unseen: train={len(train_sids)} locs, val={len(val_sids)} locs")
    return train_sids, val_sids


def split_locations_per_city(remaining_sids, sid_to_city, val_ratio=0.15, seed=42):
    """Split remaining into train and val (N% per city)."""
    rng = random.Random(seed + 1)
    city_to_sids = defaultdict(list)
    for sid in remaining_sids:
        city_to_sids[sid_to_city[sid]].append(sid)

    train_sids = set()
    val_sids = set()

    for city in sorted(city_to_sids.keys()):
        sids = sorted(city_to_sids[city])
        rng.shuffle(sids)
        n_val = max(1, round(len(sids) * val_ratio))
        val_sids.update(sids[:n_val])
        train_sids.update(sids[n_val:])

    print(f"  per_city: train={len(train_sids)} locs, val={len(val_sids)} locs")
    return train_sids, val_sids


# ─── Per-split question generation ───────────────────────────────────────────

def generate_for_split(samples, source_dirs, output_base, split_label):
    """Run Step 07 (composites) + Step 05 (questions) on a subset of samples.

    Args:
        samples: List of base sample dicts for this split only.
        source_dirs: Where to find pre-collected sat/sv images.
        output_base: Where to write generated composites.
        split_label: Label for logging (e.g., "benchmark", "train_seen_unseen").

    Returns:
        List of samples enriched with questions (hierarchical, ready to flatten).
    """
    # Import step modules (these use src/ imports internally)
    import importlib
    step07 = importlib.import_module("07_generate_composites")
    step05 = importlib.import_module("05_generate_questions")

    print(f"\n{'='*60}")
    print(f"  Generating questions for: {split_label} ({len(samples)} locations)")
    print(f"{'='*60}")

    # Prepare samples with city field at top level (Step 07 expects it)
    for s in samples:
        if "city" not in s:
            s["city"] = get_city(s)

    # Step 07: Generate composites (restricted distractor pool)
    samples = step07.run(samples, output_base=output_base, source_dirs=source_dirs)

    # Step 05: Generate questions (uses composites from step 07)
    samples = step05.run(samples, source_dirs=source_dirs)

    return samples


# ─── Flatten + filter pipeline ────────────────────────────────────────────────

def flatten_and_filter(samples, seed=42):
    """Flatten hierarchical samples into question records and apply filters.

    Returns flat question records.
    """
    # Build flat records from each sample's all_questions
    flat = []
    for sample in samples:
        all_qs = sample.get("all_questions", [])
        sid = sample["sample_id"]
        location = sample.get("location", {})
        metadata = sample.get("metadata", {})
        validation = sample.get("validation", {})
        images = sample.get("images", {})

        city = location.get("city", sample.get("city", ""))
        country = location.get("country", sample.get("country", ""))
        latitude = location.get("latitude", sample.get("lat"))
        longitude = location.get("longitude", sample.get("lon"))
        land_use = metadata.get("land_use_category", sample.get("land_use_category", ""))
        sv_count = validation.get("streetview_count", sample.get("streetview_count", 0))

        topic_counts = {}
        for q in all_qs:
            topic = q.get("topic", "unknown")
            idx = topic_counts.get(topic, 0)
            topic_counts[topic] = idx + 1

            question_id = f"{sid}_{topic}_{idx}"
            rec = {
                "question_id": question_id,
                "sample_id": sid,
                "city": city,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
                "land_use": land_use,
                "streetview_count": sv_count,
                "question": q.get("question", ""),
                "options": q.get("options", {}),
                "answer": q.get("answer", ""),
                "topic": topic,
                "difficulty": q.get("difficulty", ""),
                "generation_method": q.get("generation_method", "template"),
                "images": images,
                "sat_marked_path": q.get("sat_marked_path") or sample.get("sat_marked_path"),
                "query_stv_path": q.get("query_stv_path"),
                "query_stv_angle": q.get("query_stv_angle"),
                "option_sat_paths": q.get("option_sat_paths"),
                "mismatch_strategy": q.get("mismatch_strategy"),
                "mismatch_is_match": q.get("mismatch_is_match"),
                "mismatch_negative_stv_paths": q.get("mismatch_negative_stv_paths"),
                "mismatch_negative_stv_composite": q.get("mismatch_negative_stv_composite"),
                "composite_stv_path": q.get("composite_stv_path"),
                "composite_stv_labeled_path": q.get("composite_stv_labeled_path"),
                "option_stv_paths": q.get("option_stv_paths"),
                "option_composite_paths": q.get("option_composite_paths"),
                "stv_shown_paths": q.get("stv_shown_paths"),
                "stv_shown_composite": q.get("stv_shown_composite"),
            }
            flat.append(rec)

    print(f"  Flattened {len(samples)} locations -> {len(flat)} questions")

    # Remove excluded topics
    print(f"  Removing topics: {REMOVE_TOPICS}")
    flat, _ = remove_question_types(flat, REMOVE_TOPICS)

    # Mismatch subtypes
    print("  Splitting mismatch subtypes...")
    flat = split_mismatch_subtypes(flat)

    # Deduplicate: 1 per (sample_id, topic)
    print("  Deduplicating (1 per topic per location)...")
    flat, _ = deduplicate_per_location(flat, seed)

    return flat


# ─── Image packaging ─────────────────────────────────────────────────────────

def copy_split_images(flat_records, source_dirs, generated_dir, dest_base):
    """Copy all images referenced by flat records into dest_base/.

    Searches source_dirs for pre-collected images (sat, sv) and
    generated_dir for composites/marked/arrows.
    """
    all_paths = set()
    for rec in flat_records:
        all_paths |= collect_image_paths(rec)

    # Search order: generated dir first (has composites), then source dirs
    search_dirs = [generated_dir] + list(source_dirs)

    copied = 0
    missing = 0
    for rel_path in sorted(all_paths):
        dest = os.path.join(dest_base, rel_path)
        if os.path.exists(dest):
            copied += 1
            continue
        src = find_source(rel_path, search_dirs)
        if src:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        else:
            missing += 1

    print(f"  Images: {copied} copied, {missing} missing")
    return copied, missing


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild EODATA dataset with split-first methodology"
    )
    parser.add_argument("--input", required=True,
                        help="Path to dataset_merged.jsonl")
    parser.add_argument("--outdir", required=True,
                        help="Output directory (e.g., /mnt/hdd/EODATA_final)")
    parser.add_argument("--image-dirs", nargs="+", required=True,
                        help="Source directories containing output/images/")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation ratio for per_city strategy")
    parser.add_argument("--strategies", nargs="+",
                        choices=["seen_unseen", "per_city"],
                        default=["seen_unseen", "per_city"],
                        help="Which strategies to build")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Step 1: Load base samples ────────────────────────────────────
    print("\n[STEP 1] Loading base samples...")
    base_samples = load_base_samples(args.input)
    print(f"  Loaded {len(base_samples)} locations")

    cities_found = sorted(set(get_city(s) for s in base_samples))
    print(f"  Cities ({len(cities_found)}): {', '.join(cities_found)}")

    # ── Step 2: Filter low streetview ────────────────────────────────
    print(f"\n[STEP 2] Filtering locations with sv_count < {MIN_SV_ANGLES}...")
    before = len(base_samples)
    base_samples = [s for s in base_samples if get_sv_count(s) >= MIN_SV_ANGLES]
    print(f"  Removed {before - len(base_samples)}, remaining: {len(base_samples)}")

    # Build sid -> city lookup
    sid_to_city = {s["sample_id"]: get_city(s) for s in base_samples}
    sid_to_sample = {s["sample_id"]: s for s in base_samples}

    # ── Step 3: Split locations (benchmark) ──────────────────────────
    print(f"\n[STEP 3] Splitting benchmark locations ({BENCHMARK_RATIO*100:.0f}% per city)...")
    benchmark_sids, remaining_sids = split_locations_benchmark(
        base_samples, BENCHMARK_RATIO, args.seed
    )

    # ── Step 4: Generate benchmark dataset ───────────────────────────
    print("\n[STEP 4] Generating benchmark dataset...")
    bench_samples = [sid_to_sample[sid] for sid in sorted(benchmark_sids)]
    bench_output_base = os.path.join(args.outdir, "benchmark")
    bench_samples = generate_for_split(
        bench_samples, args.image_dirs, bench_output_base, "benchmark"
    )

    # Flatten + filter benchmark
    print("\n  Flattening benchmark...")
    bench_flat = flatten_and_filter(bench_samples, args.seed)

    # Add metadata
    for rec in bench_flat:
        rec["split"] = "benchmark"
        rec["city_type"] = "unseen" if rec["city"] in UNSEEN_CITIES else "seen"
        rec["benchmark_city_type"] = "unseen_city" if rec["city"] in UNSEEN_CITIES else "seen_city"

    bench_flat.sort(key=lambda r: r["question_id"])

    # Export benchmark
    save_jsonl(bench_flat, os.path.join(bench_output_base, "benchmark_with_answers.jsonl"))
    bench_public = [dict(r, answer=None) for r in bench_flat]
    save_jsonl(bench_public, os.path.join(bench_output_base, "benchmark_public.jsonl"))

    # Copy benchmark images
    print("\n  Packaging benchmark images...")
    copy_split_images(bench_flat, args.image_dirs, bench_output_base, bench_output_base)

    print(f"\n  Benchmark: {len(bench_flat)} questions, "
          f"{len(benchmark_sids)} locations, "
          f"{len(set(r['city'] for r in bench_flat))} cities")

    # ── Step 5: For each strategy, split remaining + generate ────────
    for strategy in args.strategies:
        print(f"\n{'#'*60}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'#'*60}")

        strategy_dir = os.path.join(args.outdir, f"splits_{strategy}")
        os.makedirs(strategy_dir, exist_ok=True)

        # Split remaining locations
        if strategy == "seen_unseen":
            train_sids, val_sids = split_locations_seen_unseen(
                remaining_sids, sid_to_city
            )
        else:
            train_sids, val_sids = split_locations_per_city(
                remaining_sids, sid_to_city, args.val_ratio, args.seed
            )

        # Generate composites + questions for train
        train_samples = [sid_to_sample[sid] for sid in sorted(train_sids)]
        train_output = os.path.join(strategy_dir, "_generated_train")
        train_samples = generate_for_split(
            train_samples, args.image_dirs, train_output,
            f"train_{strategy}"
        )

        # Generate composites + questions for val
        val_samples = [sid_to_sample[sid] for sid in sorted(val_sids)]
        val_output = os.path.join(strategy_dir, "_generated_val")
        val_samples = generate_for_split(
            val_samples, args.image_dirs, val_output,
            f"val_{strategy}"
        )

        # Flatten + filter
        print(f"\n  Flattening train ({strategy})...")
        train_flat = flatten_and_filter(train_samples, args.seed)

        print(f"\n  Flattening val ({strategy})...")
        val_flat = flatten_and_filter(val_samples, args.seed)

        # Assign city types
        for rec in train_flat:
            rec["split"] = "train"
            rec["city_type"] = "unseen" if rec["city"] in UNSEEN_CITIES else "seen"
            rec["benchmark_city_type"] = None

        for rec in val_flat:
            rec["split"] = "validation"
            rec["city_type"] = "unseen" if rec["city"] in UNSEEN_CITIES else "seen"
            rec["benchmark_city_type"] = None

        # Leak filter (seen_unseen only: remove easy mismatch referencing unseen cities)
        leak_stats = {"binary_discarded": 0, "mcq_discarded": 0}
        if strategy == "seen_unseen":
            print("\n  Applying mismatch easy leak filter on train...")
            train_flat, bd, md = filter_mismatch_leaks(train_flat, UNSEEN_CITIES)
            leak_stats = {"binary_discarded": bd, "mcq_discarded": md}

        # Adjust targets
        targets = dict(QUESTION_TARGETS)
        for topic, target in targets.items():
            if target is None:
                continue
            actual = sum(1 for r in train_flat if r["topic"] == topic)
            if actual < target:
                targets[topic] = actual

        # Downsample train
        print(f"\n  Downsampling train ({strategy})...")
        train_down = stratified_downsample(train_flat, targets, args.seed)

        # Sort
        train_down.sort(key=lambda r: r["question_id"])
        val_flat.sort(key=lambda r: r["question_id"])

        # Export JSONLs
        save_jsonl(train_down, os.path.join(strategy_dir, "train.jsonl"))
        save_jsonl(val_flat, os.path.join(strategy_dir, "validation.jsonl"))

        # Copy benchmark JSONLs for convenience
        shutil.copy2(
            os.path.join(bench_output_base, "benchmark_with_answers.jsonl"),
            os.path.join(strategy_dir, "benchmark_with_answers.jsonl"),
        )
        shutil.copy2(
            os.path.join(bench_output_base, "benchmark_public.jsonl"),
            os.path.join(strategy_dir, "benchmark_public.jsonl"),
        )

        # Package images for this strategy
        print(f"\n  Packaging images for {strategy}...")
        all_flat = train_down + val_flat
        copy_split_images(
            all_flat,
            args.image_dirs,
            # Search both train and val generated dirs for composites
            strategy_dir,  # This won't have images, but search_dirs below will
            os.path.join(strategy_dir),
        )
        # Also search the generated subdirs
        extra_sources = [train_output, val_output] + list(args.image_dirs)
        all_paths = set()
        for rec in all_flat:
            all_paths |= collect_image_paths(rec)

        missing_after = 0
        for rel_path in sorted(all_paths):
            dest = os.path.join(strategy_dir, rel_path)
            if os.path.exists(dest):
                continue
            src = find_source(rel_path, extra_sources)
            if src:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
            else:
                missing_after += 1

        if missing_after:
            print(f"  WARNING: {missing_after} images still missing for {strategy}")

        # Stats report
        print(f"\n  Generating stats report for {strategy}...")
        report_path = os.path.join(strategy_dir, "split_statistics_report.txt")
        report = generate_report(
            train_down, val_flat, bench_flat,
            UNSEEN_CITIES, SEEN_CITIES,
            leak_stats, report_path,
            strategy=strategy,
        )

        print(f"\n  {strategy}: train={len(train_down)}, val={len(val_flat)}, "
              f"bench={len(bench_flat)}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {args.outdir}")
    print(f"  Benchmark: {len(bench_flat)} questions ({len(benchmark_sids)} locations)")
    for strategy in args.strategies:
        sd = os.path.join(args.outdir, f"splits_{strategy}")
        for f in ["train.jsonl", "validation.jsonl"]:
            fp = os.path.join(sd, f)
            if os.path.exists(fp):
                n = sum(1 for _ in open(fp))
                print(f"  {strategy}/{f}: {n} records")

    print(f"\n  Benchmark is self-contained at: {bench_output_base}/")
    print(f"  Run test_image_integrity.py --root {args.outdir} to verify")


if __name__ == "__main__":
    main()
