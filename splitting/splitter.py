"""Location-level splitting into benchmark, validation, and train pools."""

import random
from collections import defaultdict


def split_locations(
    flat_records: list[dict],
    unseen_city_names: set[str],
    seen_city_names: set[str],
    benchmark_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split flat question records into benchmark, validation, and train pools.

    All splitting is done at the sample_id (location) level: all questions
    sharing a sample_id go to the same split.

    Algorithm:
        1. Group sample_ids by city.
        2. For each city (all cities, seen + unseen):
           - Shuffle sample_ids deterministically.
           - Select top 10% -> benchmark.
           - Remaining -> "remaining" pool.
        3. From remaining pool:
           - Unseen cities -> validation.
           - Seen cities -> train.

    Args:
        flat_records: Flattened question records.
        unseen_city_names: Set of canonical unseen city names.
        seen_city_names: Set of canonical seen city names.
        benchmark_ratio: Fraction of locations per city for benchmark.
        seed: Random seed.

    Returns:
        (benchmark_records, validation_records, train_records)
    """
    rng = random.Random(seed)

    # Group sample_ids by city
    city_to_sids = defaultdict(set)
    for rec in flat_records:
        city_to_sids[rec["city"]].add(rec["sample_id"])

    # Index: sample_id -> list of question records
    sid_to_records = defaultdict(list)
    for rec in flat_records:
        sid_to_records[rec["sample_id"]].append(rec)

    benchmark_sids = set()
    remaining_sids = set()

    print(f"\n  {'City':<25s} {'Total':>6s} {'Bench':>6s} {'Remain':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")

    for city in sorted(city_to_sids.keys()):
        sids = sorted(city_to_sids[city])  # Sort for reproducibility before shuffle
        rng.shuffle(sids)
        n_bench = max(1, round(len(sids) * benchmark_ratio))
        bench = set(sids[:n_bench])
        remain = set(sids[n_bench:])
        benchmark_sids |= bench
        remaining_sids |= remain
        print(f"  {city:<25s} {len(sids):>6d} {len(bench):>6d} {len(remain):>6d}")

    # Split remaining into validation (unseen) and train (seen)
    validation_sids = set()
    train_sids = set()
    for sid in remaining_sids:
        city = sid_to_records[sid][0]["city"]
        if city in unseen_city_names:
            validation_sids.add(sid)
        elif city in seen_city_names:
            train_sids.add(sid)
        else:
            # City not in either list — default to train with warning
            print(f"  WARNING: City '{city}' not in seen or unseen lists, assigning to train")
            train_sids.add(sid)

    # Build output lists and tag records
    benchmark_records = []
    for sid in benchmark_sids:
        city = sid_to_records[sid][0]["city"]
        city_type = "unseen" if city in unseen_city_names else "seen"
        for rec in sid_to_records[sid]:
            rec["benchmark_city_type"] = f"{city_type}_city"
            benchmark_records.append(rec)

    validation_records = []
    for sid in validation_sids:
        for rec in sid_to_records[sid]:
            validation_records.append(rec)

    train_records = []
    for sid in train_sids:
        for rec in sid_to_records[sid]:
            train_records.append(rec)

    # Summary
    bench_seen = sum(1 for r in benchmark_records if r.get("benchmark_city_type") == "seen_city")
    bench_unseen = sum(1 for r in benchmark_records if r.get("benchmark_city_type") == "unseen_city")

    print(f"\n  Total benchmark locations: {len(benchmark_sids)}")
    print(f"  Total benchmark questions: {len(benchmark_records)}")
    print(f"    From seen cities: {bench_seen} questions")
    print(f"    From unseen cities: {bench_unseen} questions")
    print(f"  Validation locations: {len(validation_sids)}, questions: {len(validation_records)}")
    print(f"  Train locations: {len(train_sids)}, questions: {len(train_records)}")

    return benchmark_records, validation_records, train_records
