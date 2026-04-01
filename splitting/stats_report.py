"""Statistics report generation and integrity checks for dataset splits."""

from collections import Counter, defaultdict


def generate_report(
    train: list[dict],
    validation: list[dict],
    benchmark: list[dict],
    unseen_city_names: set[str],
    seen_city_names: set[str],
    leak_stats: dict,
    output_path: str,
) -> str:
    """Generate a comprehensive statistics report with integrity checks.

    Args:
        train: Train split records.
        validation: Validation split records.
        benchmark: Benchmark split records.
        unseen_city_names: Set of unseen city names.
        seen_city_names: Set of seen city names.
        leak_stats: Dict with leak filter statistics.
        output_path: Path to write the report.

    Returns:
        Report text.
    """
    lines = []

    def w(text=""):
        lines.append(text)

    separator = "=" * 60

    w(separator)
    w("SPLIT SUMMARY")
    w(separator)

    # --- TRAIN ---
    w()
    _write_split_section(w, "TRAIN", train, seen_city_names | unseen_city_names)

    # --- VALIDATION ---
    w()
    _write_split_section(w, "VALIDATION", validation, seen_city_names | unseen_city_names)

    # Per city breakdown for validation
    w()
    w("  Per city:")
    city_counts = Counter(r["city"] for r in validation)
    for city in sorted(city_counts.keys()):
        w(f"    {city:<25s} {city_counts[city]:>6d} questions")

    # --- BENCHMARK ---
    w()
    _write_split_section(w, "BENCHMARK", benchmark, seen_city_names | unseen_city_names)

    bench_seen = [r for r in benchmark if r.get("benchmark_city_type") == "seen_city"]
    bench_unseen = [r for r in benchmark if r.get("benchmark_city_type") == "unseen_city"]
    w(f"  Seen city questions   : {len(bench_seen)} ({_pct(len(bench_seen), len(benchmark))})")
    w(f"  Unseen city questions : {len(bench_unseen)} ({_pct(len(bench_unseen), len(benchmark))})")

    w()
    w("  Per city:")
    city_counts = Counter(r["city"] for r in benchmark)
    for city in sorted(city_counts.keys()):
        city_type = "unseen" if city in unseen_city_names else "seen"
        w(f"    {city:<25s} {city_counts[city]:>6d} questions  [{city_type}]")

    # --- LEAK AUDIT ---
    w()
    w(separator)
    w("LEAK AUDIT")
    w(separator)
    w(f"  mismatch_binary_easy discarded from train (unseen distractor): {leak_stats.get('binary_discarded', 0)}")
    w(f"  mismatch_mcq_easy discarded from train (unseen distractor)  : {leak_stats.get('mcq_discarded', 0)}")

    # Verify no unseen city distractors remain in train mismatch_easy
    train_binary_easy_unseen = _check_remaining_leaks(train, "mismatch_binary_easy", unseen_city_names)
    train_mcq_easy_unseen = _check_remaining_leaks(train, "mismatch_mcq_easy", unseen_city_names)
    w(f"  No unseen city in train mismatch_binary_easy: {'YES' if train_binary_easy_unseen == 0 else 'NO (' + str(train_binary_easy_unseen) + ' found)'}")
    w(f"  No unseen city in train mismatch_mcq_easy   : {'YES' if train_mcq_easy_unseen == 0 else 'NO (' + str(train_mcq_easy_unseen) + ' found)'}")

    # --- INTEGRITY CHECKS ---
    w()
    w(separator)
    w("INTEGRITY CHECKS")
    w(separator)

    train_sids = set(r["sample_id"] for r in train)
    val_sids = set(r["sample_id"] for r in validation)
    bench_sids = set(r["sample_id"] for r in benchmark)

    overlap_tv = train_sids & val_sids
    overlap_tb = train_sids & bench_sids
    overlap_vb = val_sids & bench_sids

    w(f"  Location overlap train <-> validation : {len(overlap_tv)} {'(PASS)' if len(overlap_tv) == 0 else '(FAIL)'}")
    w(f"  Location overlap train <-> benchmark  : {len(overlap_tb)} {'(PASS)' if len(overlap_tb) == 0 else '(FAIL)'}")
    w(f"  Location overlap val <-> benchmark    : {len(overlap_vb)} {'(PASS)' if len(overlap_vb) == 0 else '(FAIL)'}")

    all_cities = set(r["city"] for r in train + validation + benchmark)
    bench_cities = set(r["city"] for r in benchmark)
    val_cities = set(r["city"] for r in validation)
    train_cities = set(r["city"] for r in train)

    w(f"  All dataset cities in benchmark        : {len(bench_cities)}/{len(all_cities)} cities {'(PASS)' if bench_cities == all_cities else '(FAIL: missing ' + str(all_cities - bench_cities) + ')'}")

    # Check unseen cities: only verify those present in the input data
    unseen_in_data = unseen_city_names & all_cities
    unseen_in_val_or_bench = (val_cities | bench_cities) & unseen_city_names
    unseen_not_in_data = unseen_city_names - all_cities
    if unseen_not_in_data:
        w(f"  Unseen cities not in input data        : {sorted(unseen_not_in_data)} (OK - not in merged dataset)")
    w(f"  All unseen cities (in data) in val     : {'YES (PASS)' if unseen_in_data <= unseen_in_val_or_bench else 'NO (FAIL: missing ' + str(unseen_in_data - unseen_in_val_or_bench) + ')'}")

    # Check no unseen city locations in train (location-level, not question-level via distractors)
    train_unseen = train_cities & unseen_city_names
    w(f"  No unseen city locations in train      : {'YES (PASS)' if len(train_unseen) == 0 else 'NO (FAIL: ' + str(train_unseen) + ')'}")

    # SV count check
    all_records = train + validation + benchmark
    sv_violations = sum(1 for r in all_records if r.get("streetview_count", 0) != 4)
    w(f"  sv_angle_count == 4 for all records    : {'YES (PASS)' if sv_violations == 0 else 'NO (FAIL: ' + str(sv_violations) + ' violations)'}")

    # Water proximity check
    wp_count = sum(1 for r in all_records if r["topic"] == "water_proximity")
    w(f"  water_proximity count in any split     : {wp_count} {'(PASS)' if wp_count == 0 else '(FAIL)'}")

    w()
    w(separator)

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report


def _write_split_section(w, name: str, records: list[dict], all_city_names: set):
    """Write a split's statistics section."""
    w(f"[{name}]")
    locations = set(r["sample_id"] for r in records)
    cities = sorted(set(r["city"] for r in records))
    w(f"  Total questions     : {len(records)}")
    w(f"  Total locations     : {len(locations)}")
    w(f"  Cities              : {len(cities)} ({', '.join(cities)})")

    # Per question type
    w()
    w("  Per question type:")
    topic_counts = Counter(r["topic"] for r in records)
    for topic in sorted(topic_counts.keys()):
        count = topic_counts[topic]
        w(f"    {topic:<30s} {count:>6d}  ({_pct(count, len(records))})")

    # Per difficulty
    w()
    w("  Per difficulty:")
    diff_counts = Counter(r["difficulty"] for r in records)
    for diff in ["easy", "medium", "hard"]:
        count = diff_counts.get(diff, 0)
        w(f"    {diff:<10s} {count:>6d}  ({_pct(count, len(records))})")

    # Per land use
    w()
    w("  Per land use:")
    lu_counts = Counter(r.get("land_use", "unknown") for r in records)
    for lu in sorted(lu_counts.keys()):
        count = lu_counts[lu]
        w(f"    {lu:<20s} {count:>6d}  ({_pct(count, len(records))})")

    # Per answer key
    w()
    w("  Per answer key:")
    ans_counts = Counter(r["answer"] for r in records)
    for ans in sorted(ans_counts.keys()):
        count = ans_counts[ans]
        w(f"    {ans:<5s} {count:>6d}  ({_pct(count, len(records))})")


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * count / total:.1f}%"


def _check_remaining_leaks(records, topic, unseen_city_names):
    """Count remaining records of given topic that reference unseen cities in distractors."""
    from splitting.flatten import extract_sid_from_stv_path, sid_to_city_name

    count = 0
    for rec in records:
        if rec["topic"] != topic:
            continue

        if topic == "mismatch_binary_easy":
            if rec.get("mismatch_is_match"):
                continue
            neg_paths = rec.get("mismatch_negative_stv_paths") or []
            for path in neg_paths:
                sid = extract_sid_from_stv_path(path)
                city = sid_to_city_name(sid)
                if city and city in unseen_city_names:
                    count += 1
                    break

        elif topic == "mismatch_mcq_easy":
            option_stv = rec.get("option_stv_paths") or {}
            correct = rec.get("answer", "")
            for opt_key, paths in option_stv.items():
                if opt_key == correct:
                    continue
                if paths:
                    sid = extract_sid_from_stv_path(paths[0])
                    city = sid_to_city_name(sid)
                    if city and city in unseen_city_names:
                        count += 1
                        break

    return count
