"""Filtering and subtype assignment for the dataset splitting pipeline."""

from splitting.flatten import (
    extract_sid_from_stv_path,
    sid_to_city_name,
    CITY_KEY_TO_NAME,
)


def filter_low_streetview(records: list[dict], min_count: int = 4) -> tuple[list, int, int]:
    """Remove locations where validation.streetview_count < min_count.

    Operates on location-level records (before flattening).

    Returns:
        (filtered_records, locations_removed, questions_removed)
    """
    kept = []
    removed_locs = 0
    removed_qs = 0
    for rec in records:
        sv_count = rec.get("validation", {}).get("streetview_count", 0)
        if sv_count >= min_count:
            kept.append(rec)
        else:
            removed_locs += 1
            removed_qs += len(rec.get("questions", []))

    print(f"  Locations removed (sv < {min_count}): {removed_locs}")
    print(f"  Questions removed with them: {removed_qs}")
    print(f"  Remaining locations: {len(kept)}")
    return kept, removed_locs, removed_qs


def remove_question_types(flat_records: list[dict], types: list[str]) -> tuple[list, int]:
    """Remove questions with specified topics.

    Args:
        flat_records: Flattened question records.
        types: List of topic names to remove (e.g., ["water_proximity"]).

    Returns:
        (filtered_records, removed_count)
    """
    types_set = set(types)
    kept = []
    removed = 0
    for rec in flat_records:
        if rec["topic"] in types_set:
            removed += 1
        else:
            kept.append(rec)

    for t in types:
        print(f"  Questions removed ({t}): {sum(1 for r in flat_records if r['topic'] == t)}")
    print(f"  Remaining questions: {len(kept)}")
    return kept, removed


def split_mismatch_subtypes(flat_records: list[dict]) -> list[dict]:
    """Rename mismatch topics to include difficulty suffix.

    mismatch_binary + hard -> mismatch_binary_hard
    mismatch_binary + easy/medium -> mismatch_binary_easy
    mismatch_mcq + hard -> mismatch_mcq_hard
    mismatch_mcq + easy/medium -> mismatch_mcq_easy
    """
    counts = {
        "mismatch_binary_easy": 0,
        "mismatch_binary_hard": 0,
        "mismatch_mcq_easy": 0,
        "mismatch_mcq_hard": 0,
    }

    for rec in flat_records:
        if rec["topic"] in ("mismatch_binary", "mismatch_mcq"):
            suffix = "hard" if rec["difficulty"] == "hard" else "easy"
            new_topic = f"{rec['topic']}_{suffix}"
            rec["topic"] = new_topic
            counts[new_topic] = counts.get(new_topic, 0) + 1

    print("  Mismatch subtype counts:")
    for subtype, count in sorted(counts.items()):
        print(f"    {subtype}: {count}")

    return flat_records


def _get_distractor_cities_binary(rec: dict) -> set[str]:
    """Extract distractor city names from a mismatch_binary question.

    For NO variants (mismatch_is_match == False), the distractor STV paths
    contain the negative sample_id from which we infer the city.
    For YES variants, no distractor city (shows own location).
    """
    if rec.get("mismatch_is_match") is True:
        # YES variant: streetview shown is from the same location, no distractor
        return set()

    neg_paths = rec.get("mismatch_negative_stv_paths") or []
    cities = set()
    for path in neg_paths:
        sid = extract_sid_from_stv_path(path)
        city = sid_to_city_name(sid)
        if city:
            cities.add(city)
    return cities


def _get_distractor_cities_mcq(rec: dict) -> set[str]:
    """Extract distractor city names from a mismatch_mcq question.

    For each option that is NOT the correct answer, extract the sample_id
    from the streetview paths and infer the city.
    """
    option_stv = rec.get("option_stv_paths") or {}
    correct_answer = rec.get("answer", "")
    cities = set()

    for opt_key, paths in option_stv.items():
        if opt_key == correct_answer:
            continue  # Skip the correct answer (anchor location)
        if paths:
            sid = extract_sid_from_stv_path(paths[0])
            city = sid_to_city_name(sid)
            if city:
                cities.add(city)
    return cities


def filter_mismatch_leaks(
    train_records: list[dict], unseen_city_names: set[str]
) -> tuple[list, int, int]:
    """Remove mismatch_easy train questions that reference unseen city distractors.

    Only applies to mismatch_binary_easy and mismatch_mcq_easy.
    Hard mismatch questions use same-city distractors and don't need filtering.

    Args:
        train_records: Flat question records assigned to train split.
        unseen_city_names: Set of canonical unseen city names.

    Returns:
        (filtered_records, binary_easy_discarded, mcq_easy_discarded)
    """
    kept = []
    binary_discarded = 0
    mcq_discarded = 0

    for rec in train_records:
        topic = rec["topic"]

        if topic == "mismatch_binary_easy":
            distractor_cities = _get_distractor_cities_binary(rec)
            if distractor_cities & unseen_city_names:
                binary_discarded += 1
                continue

        elif topic == "mismatch_mcq_easy":
            distractor_cities = _get_distractor_cities_mcq(rec)
            if distractor_cities & unseen_city_names:
                mcq_discarded += 1
                continue

        kept.append(rec)

    print(f"  mismatch_binary_easy discarded (leak): {binary_discarded}")
    print(f"  mismatch_binary_easy remaining: {sum(1 for r in kept if r['topic'] == 'mismatch_binary_easy')}")
    print(f"  mismatch_mcq_easy discarded (leak): {mcq_discarded}")
    print(f"  mismatch_mcq_easy remaining: {sum(1 for r in kept if r['topic'] == 'mismatch_mcq_easy')}")

    return kept, binary_discarded, mcq_discarded
