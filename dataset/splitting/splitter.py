"""Location-level splitting into benchmark, validation, and train pools."""

import random
from collections import defaultdict


def _build_indexes(flat_records):
    """Build city->sample_ids and sample_id->records indexes."""
    city_to_sids = defaultdict(set)
    sid_to_records = defaultdict(list)
    for rec in flat_records:
        city_to_sids[rec["city"]].add(rec["sample_id"])
        sid_to_records[rec["sample_id"]].append(rec)
    return city_to_sids, sid_to_records


def extract_benchmark(
    flat_records: list[dict],
    unseen_city_names: set[str],
    benchmark_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list]:
    """Extract benchmark split (10% per city) and return remainder.

    This is strategy-independent — always called first, identically,
    so benchmark is the same regardless of train/val strategy.

    Args:
        flat_records: Flattened question records.
        unseen_city_names: Set of unseen city names (for tagging only).
        benchmark_ratio: Fraction of locations per city for benchmark.
        seed: Random seed.

    Returns:
        (benchmark_records, remaining_records)
    """
    rng = random.Random(seed)
    city_to_sids, sid_to_records = _build_indexes(flat_records)

    benchmark_sids = set()
    remaining_sids = set()

    print(f"\n  {'City':<25s} {'Total':>6s} {'Bench':>6s} {'Remain':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")

    for city in sorted(city_to_sids.keys()):
        sids = sorted(city_to_sids[city])
        rng.shuffle(sids)
        n_bench = max(1, round(len(sids) * benchmark_ratio))
        benchmark_sids.update(sids[:n_bench])
        remaining_sids.update(sids[n_bench:])
        print(f"  {city:<25s} {len(sids):>6d} {n_bench:>6d} {len(sids) - n_bench:>6d}")

    # Build output lists and tag benchmark records
    benchmark_records = []
    for sid in benchmark_sids:
        city = sid_to_records[sid][0]["city"]
        city_type = "unseen_city" if city in unseen_city_names else "seen_city"
        for rec in sid_to_records[sid]:
            rec["benchmark_city_type"] = city_type
            benchmark_records.append(rec)

    remaining_records = []
    for sid in remaining_sids:
        for rec in sid_to_records[sid]:
            remaining_records.append(rec)

    bench_seen = sum(1 for r in benchmark_records if r.get("benchmark_city_type") == "seen_city")
    bench_unseen = sum(1 for r in benchmark_records if r.get("benchmark_city_type") == "unseen_city")

    print(f"\n  Total benchmark locations: {len(benchmark_sids)}")
    print(f"  Total benchmark questions: {len(benchmark_records)}")
    print(f"    From seen cities: {bench_seen} questions")
    print(f"    From unseen cities: {bench_unseen} questions")
    print(f"  Remaining locations: {len(remaining_sids)}, questions: {len(remaining_records)}")

    return benchmark_records, remaining_records


def split_seen_unseen(
    remaining_records: list[dict],
    unseen_city_names: set[str],
    seen_city_names: set[str],
) -> tuple[list, list]:
    """Split remainder by seen/unseen cities.

    Unseen cities -> validation, seen cities -> train.

    Returns:
        (validation_records, train_records)
    """
    validation_records = []
    train_records = []

    for rec in remaining_records:
        if rec["city"] in unseen_city_names:
            validation_records.append(rec)
        elif rec["city"] in seen_city_names:
            train_records.append(rec)
        else:
            print(f"  WARNING: City '{rec['city']}' not in seen or unseen lists, assigning to train")
            train_records.append(rec)

    val_locs = len(set(r["sample_id"] for r in validation_records))
    train_locs = len(set(r["sample_id"] for r in train_records))
    print(f"  Validation: {val_locs} locations, {len(validation_records)} questions")
    print(f"  Train: {train_locs} locations, {len(train_records)} questions")

    return validation_records, train_records


def split_per_city(
    remaining_records: list[dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list]:
    """Split remainder by sampling N% of locations per city for validation.

    Every city contributes to both train and validation proportionally.

    Returns:
        (validation_records, train_records)
    """
    rng = random.Random(seed + 1)  # Different seed offset to not correlate with benchmark shuffle
    city_to_sids, sid_to_records = _build_indexes(remaining_records)

    validation_sids = set()
    train_sids = set()

    print(f"\n  {'City':<25s} {'Remain':>6s} {'Val':>6s} {'Train':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")

    for city in sorted(city_to_sids.keys()):
        sids = sorted(city_to_sids[city])
        rng.shuffle(sids)
        n_val = max(1, round(len(sids) * val_ratio))
        validation_sids.update(sids[:n_val])
        train_sids.update(sids[n_val:])
        print(f"  {city:<25s} {len(sids):>6d} {n_val:>6d} {len(sids) - n_val:>6d}")

    validation_records = []
    for sid in validation_sids:
        validation_records.extend(sid_to_records[sid])

    train_records = []
    for sid in train_sids:
        train_records.extend(sid_to_records[sid])

    print(f"\n  Validation: {len(validation_sids)} locations, {len(validation_records)} questions")
    print(f"  Train: {len(train_sids)} locations, {len(train_records)} questions")

    return validation_records, train_records
