#!/usr/bin/env python3
"""
Image integrity tester for EODATA_splits.

Verifies that every image path referenced in every JSONL file resolves to
an actual file on disk.  Covers:
  - benchmark/          (self-contained, own images/)
  - splits_seen_unseen/ (train + val use all_data/images/)
  - splits_per_city/    (train + val use all_data/images/)

Also checks:
  - Benchmark JSONLs are identical across all three locations
  - benchmark_public has no answer leakage
  - No location (sample_id) overlap between train/val/benchmark per strategy
  - No duplicate question_ids per strategy

Usage:
    python splitting/test_image_integrity.py --root /mnt/usb/EODATA_splits
    python splitting/test_image_integrity.py --root /mnt/usb/EODATA_splits --verbose
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ERROR: {path} line {i}: {e}")
                sys.exit(1)
    return records


def extract_image_paths(record: dict) -> set[str]:
    """Recursively extract all strings starting with 'images/' from a record."""
    paths = set()

    def walk(obj):
        if isinstance(obj, str):
            if obj.startswith("images/"):
                paths.add(obj)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(record)
    return paths


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_path(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) >= 2 else "unknown"


# ── Test functions ───────────────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def ok(self, msg: str):
        self.passed += 1
        self.details.append(("PASS", msg))

    def fail(self, msg: str):
        self.failed += 1
        self.details.append(("FAIL", msg))

    def warn(self, msg: str):
        self.warnings += 1
        self.details.append(("WARN", msg))


def test_jsonl_valid(results: TestResult, path: str) -> list[dict]:
    """Test that a JSONL file is valid and has required fields."""
    name = os.path.basename(path)
    records = load_jsonl(path)

    if len(records) == 0:
        results.fail(f"{name}: file is empty")
        return records

    required = {"question_id", "sample_id", "city", "topic", "answer", "question", "options"}
    missing_fields = set()
    for rec in records:
        for field in required:
            if field not in rec:
                missing_fields.add(field)

    if missing_fields:
        results.fail(f"{name}: missing required fields: {missing_fields}")
    else:
        results.ok(f"{name}: {len(records)} records, all required fields present")

    return records


def test_images_exist(
    results: TestResult,
    records: list[dict],
    image_base: str,
    label: str,
    verbose: bool = False,
) -> tuple[int, int]:
    """Test that all image paths in records resolve to files under image_base."""
    all_paths = set()
    for rec in records:
        all_paths.update(extract_image_paths(rec))

    missing = []
    for p in sorted(all_paths):
        full = os.path.join(image_base, p)
        if not os.path.isfile(full):
            missing.append(p)

    if len(missing) == 0:
        results.ok(f"{label}: all {len(all_paths)} image refs exist")
    else:
        by_type = defaultdict(list)
        for p in missing:
            by_type[classify_path(p)].append(p)

        breakdown = ", ".join(f"{t}: {len(ps)}" for t, ps in sorted(by_type.items()))
        results.fail(f"{label}: {len(missing)}/{len(all_paths)} images missing ({breakdown})")

        if verbose:
            for p in missing[:20]:
                print(f"      MISSING: {p}")
            if len(missing) > 20:
                print(f"      ... and {len(missing) - 20} more")

    return len(all_paths), len(missing)


def test_no_answer_leakage(results: TestResult, records: list[dict], label: str):
    """Test that all answer fields are None in public benchmark."""
    non_null = sum(1 for r in records if r.get("answer") is not None)
    if non_null == 0:
        results.ok(f"{label}: all {len(records)} answers are null (no leakage)")
    else:
        results.fail(f"{label}: {non_null} records have non-null answers!")


def test_all_answers_present(results: TestResult, records: list[dict], label: str):
    """Test that all answer fields are non-None in answers benchmark."""
    null_count = sum(1 for r in records if r.get("answer") is None)
    if null_count == 0:
        results.ok(f"{label}: all {len(records)} records have answers")
    else:
        results.fail(f"{label}: {null_count} records have null answers!")


def test_no_location_overlap(
    results: TestResult,
    train: list[dict],
    val: list[dict],
    bench: list[dict],
    label: str,
):
    """Test that no sample_id appears in more than one split."""
    t_sids = set(r["sample_id"] for r in train)
    v_sids = set(r["sample_id"] for r in val)
    b_sids = set(r["sample_id"] for r in bench)

    tv = t_sids & v_sids
    tb = t_sids & b_sids
    vb = v_sids & b_sids

    if len(tv) == 0:
        results.ok(f"{label}: train/val location overlap = 0")
    else:
        results.fail(f"{label}: train/val share {len(tv)} locations: {sorted(tv)[:5]}...")

    if len(tb) == 0:
        results.ok(f"{label}: train/bench location overlap = 0")
    else:
        results.fail(f"{label}: train/bench share {len(tb)} locations: {sorted(tb)[:5]}...")

    if len(vb) == 0:
        results.ok(f"{label}: val/bench location overlap = 0")
    else:
        results.fail(f"{label}: val/bench share {len(vb)} locations: {sorted(vb)[:5]}...")


def test_no_duplicate_qids(
    results: TestResult,
    train: list[dict],
    val: list[dict],
    bench: list[dict],
    label: str,
):
    """Test that question_ids are unique across the entire strategy."""
    all_qids = []
    for split in [train, val, bench]:
        all_qids.extend(r["question_id"] for r in split)

    dupes = len(all_qids) - len(set(all_qids))
    if dupes == 0:
        results.ok(f"{label}: {len(all_qids)} question_ids, all unique")
    else:
        results.fail(f"{label}: {dupes} duplicate question_ids!")


def test_benchmark_identical(results: TestResult, path_a: str, path_b: str, label: str):
    """Test that two benchmark files are byte-identical."""
    md5_a = md5_file(path_a)
    md5_b = md5_file(path_b)
    name_a = "/".join(path_a.split("/")[-2:])
    name_b = "/".join(path_b.split("/")[-2:])
    if md5_a == md5_b:
        results.ok(f"{label}: {name_a} == {name_b}")
    else:
        results.fail(f"{label}: {name_a} ({md5_a}) != {name_b} ({md5_b})")


def test_unseen_cities_not_in_train(results: TestResult, train: list[dict], label: str):
    """Test that unseen cities do not appear in the seen_unseen train split."""
    UNSEEN = {
        "Istanbul", "Moscow", "Chicago", "Seoul",
        "Rio de Janeiro", "Cape Town", "Sydney", "Singapore",
    }
    train_cities = set(r["city"] for r in train)
    leaked = train_cities & UNSEEN
    if len(leaked) == 0:
        results.ok(f"{label}: no unseen cities in train ({len(train_cities)} cities)")
    else:
        results.fail(f"{label}: unseen cities found in train: {leaked}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test EODATA_splits image integrity and consistency")
    parser.add_argument("--root", required=True, help="Path to EODATA_splits directory")
    parser.add_argument("--verbose", action="store_true", help="Print individual missing file paths")
    args = parser.parse_args()

    root = args.root
    results = TestResult()
    total_refs = 0
    total_missing = 0

    # Paths
    bench_dir = os.path.join(root, "benchmark")
    all_data_dir = os.path.join(root, "all_data")
    su_dir = os.path.join(root, "splits_seen_unseen")
    pc_dir = os.path.join(root, "splits_per_city")

    print("=" * 70)
    print("EODATA SPLITS — INTEGRITY TEST")
    print("=" * 70)

    # ── 1. Benchmark directory ───────────────────────────────────────────
    print("\n[1] BENCHMARK DIRECTORY")
    bench_pub = test_jsonl_valid(results, os.path.join(bench_dir, "benchmark_public.jsonl"))
    bench_ans = test_jsonl_valid(results, os.path.join(bench_dir, "benchmark_with_answers.jsonl"))

    test_no_answer_leakage(results, bench_pub, "benchmark_public")
    test_all_answers_present(results, bench_ans, "benchmark_with_answers")

    refs, miss = test_images_exist(results, bench_ans, bench_dir, "benchmark images", args.verbose)
    total_refs += refs
    total_missing += miss

    # ── 2. Benchmark consistency across locations ────────────────────────
    print("\n[2] BENCHMARK CONSISTENCY")
    test_benchmark_identical(
        results,
        os.path.join(bench_dir, "benchmark_public.jsonl"),
        os.path.join(su_dir, "benchmark_public.jsonl"),
        "bench_public vs seen_unseen",
    )
    test_benchmark_identical(
        results,
        os.path.join(bench_dir, "benchmark_with_answers.jsonl"),
        os.path.join(su_dir, "benchmark_with_answers.jsonl"),
        "bench_answers vs seen_unseen",
    )
    test_benchmark_identical(
        results,
        os.path.join(su_dir, "benchmark_public.jsonl"),
        os.path.join(pc_dir, "benchmark_public.jsonl"),
        "seen_unseen vs per_city public",
    )
    test_benchmark_identical(
        results,
        os.path.join(su_dir, "benchmark_with_answers.jsonl"),
        os.path.join(pc_dir, "benchmark_with_answers.jsonl"),
        "seen_unseen vs per_city answers",
    )

    # ── 3. splits_seen_unseen ────────────────────────────────────────────
    print("\n[3] SPLITS_SEEN_UNSEEN")
    su_train = test_jsonl_valid(results, os.path.join(su_dir, "train.jsonl"))
    su_val = test_jsonl_valid(results, os.path.join(su_dir, "validation.jsonl"))
    su_bench = test_jsonl_valid(results, os.path.join(su_dir, "benchmark_with_answers.jsonl"))

    test_no_location_overlap(results, su_train, su_val, su_bench, "seen_unseen")
    test_no_duplicate_qids(results, su_train, su_val, su_bench, "seen_unseen")
    test_unseen_cities_not_in_train(results, su_train, "seen_unseen")

    refs, miss = test_images_exist(results, su_train, all_data_dir, "seen_unseen/train images", args.verbose)
    total_refs += refs
    total_missing += miss

    refs, miss = test_images_exist(results, su_val, all_data_dir, "seen_unseen/val images", args.verbose)
    total_refs += refs
    total_missing += miss

    # ── 4. splits_per_city ───────────────────────────────────────────────
    print("\n[4] SPLITS_PER_CITY")
    pc_train = test_jsonl_valid(results, os.path.join(pc_dir, "train.jsonl"))
    pc_val = test_jsonl_valid(results, os.path.join(pc_dir, "validation.jsonl"))
    pc_bench = test_jsonl_valid(results, os.path.join(pc_dir, "benchmark_with_answers.jsonl"))

    test_no_location_overlap(results, pc_train, pc_val, pc_bench, "per_city")
    test_no_duplicate_qids(results, pc_train, pc_val, pc_bench, "per_city")

    refs, miss = test_images_exist(results, pc_train, all_data_dir, "per_city/train images", args.verbose)
    total_refs += refs
    total_missing += miss

    refs, miss = test_images_exist(results, pc_val, all_data_dir, "per_city/val images", args.verbose)
    total_refs += refs
    total_missing += miss

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for status, msg in results.details:
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
        print(f"  [{icon}] {msg}")

    print(f"\n  Total image refs checked : {total_refs}")
    print(f"  Total images missing     : {total_missing}")
    print(f"\n  PASSED  : {results.passed}")
    print(f"  FAILED  : {results.failed}")
    print(f"  WARNINGS: {results.warnings}")

    if results.failed > 0:
        print(f"\n  RESULT: FAIL ({results.failed} failures)")
        sys.exit(1)
    else:
        print(f"\n  RESULT: ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
