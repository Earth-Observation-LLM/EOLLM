"""Stratified downsampling for the train split."""

import math
import random
from collections import defaultdict


def stratified_downsample(
    train_records: list[dict],
    targets: dict[str, int | None],
    seed: int = 42,
) -> list[dict]:
    """Downsample train records per topic using stratified sampling.

    For each topic:
        - If target is None: keep all (e.g., building_height).
        - If pool <= target: keep all, print warning.
        - Otherwise: stratified sample by (city, land_use, difficulty, answer).

    Args:
        train_records: Flat question records in the train split.
        targets: Dict mapping topic -> target count (None = keep all).
        seed: Random seed.

    Returns:
        Downsampled list of train records.
    """
    rng = random.Random(seed)

    # Group records by topic
    by_topic = defaultdict(list)
    for rec in train_records:
        by_topic[rec["topic"]].append(rec)

    result = []

    print(f"\n  {'Topic':<30s} {'Pool':>6s} {'Target':>7s} {'Final':>6s}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*6}")

    for topic in sorted(set(list(targets.keys()) + list(by_topic.keys()))):
        pool = by_topic.get(topic, [])
        target = targets.get(topic)

        if not pool:
            if target is not None:
                print(f"  {topic:<30s} {0:>6d} {str(target):>7s} {0:>6d}  WARNING: no questions")
            continue

        # Keep all if target is None or pool is small enough
        if target is None:
            result.extend(pool)
            print(f"  {topic:<30s} {len(pool):>6d} {'all':>7s} {len(pool):>6d}")
            continue

        if len(pool) <= target:
            result.extend(pool)
            print(f"  {topic:<30s} {len(pool):>6d} {target:>7d} {len(pool):>6d}  WARNING: under target")
            continue

        # Stratified sampling
        sampled = _stratified_sample(pool, target, rng)
        result.extend(sampled)
        print(f"  {topic:<30s} {len(pool):>6d} {target:>7d} {len(sampled):>6d}")

    # Handle topics not in targets (keep all)
    for topic, recs in by_topic.items():
        if topic not in targets:
            result.extend(recs)
            print(f"  {topic:<30s} {len(recs):>6d} {'N/A':>7s} {len(recs):>6d}  (no target defined)")

    print(f"\n  Total train after downsampling: {len(result)}")
    return result


def _stratified_sample(
    pool: list[dict], target: int, rng: random.Random
) -> list[dict]:
    """Perform iterative proportional stratified sampling.

    Groups by (city, land_use, difficulty, answer) and samples proportionally.
    """
    # Group by stratification key
    groups = defaultdict(list)
    for rec in pool:
        key = (
            rec.get("city", ""),
            rec.get("land_use", ""),
            rec.get("difficulty", ""),
            rec.get("answer", ""),
        )
        groups[key].append(rec)

    total_pool = len(pool)
    sampled = []
    remaining_target = target

    # Sort groups for reproducibility
    sorted_keys = sorted(groups.keys())

    # First pass: allocate proportionally (floor)
    allocations = {}
    for key in sorted_keys:
        group = groups[key]
        proportion = len(group) / total_pool
        alloc = min(math.floor(proportion * target), len(group))
        allocations[key] = alloc

    # Distribute remainder
    allocated_total = sum(allocations.values())
    remainder = target - allocated_total

    if remainder > 0:
        # Sort by fractional part descending (largest shortfall first)
        fractional = []
        for key in sorted_keys:
            group = groups[key]
            proportion = len(group) / total_pool
            ideal = proportion * target
            frac = ideal - allocations[key]
            headroom = len(group) - allocations[key]
            if headroom > 0:
                fractional.append((frac, key))
        fractional.sort(key=lambda x: -x[0])

        for _, key in fractional:
            if remainder <= 0:
                break
            headroom = len(groups[key]) - allocations[key]
            add = min(1, headroom)
            allocations[key] += add
            remainder -= add

    # Sample from each group
    for key in sorted_keys:
        group = groups[key]
        n = allocations[key]
        if n >= len(group):
            sampled.extend(group)
        else:
            sampled.extend(rng.sample(group, n))

    return sampled
