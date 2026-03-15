"""
Step 5: Generate template-based MCQs from OSM metadata and cross-view alignment.
City-agnostic — questions derived from universal OSM data and coordinate geometry.
"""

import os
import json
import random
from collections import Counter
from question_templates import QUESTION_TEMPLATES
from utils import bearing_to_quadrant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shuffle_options(correct, distractors, sample_id="", topic=""):
    """Shuffle options and return (options_dict, answer_key).

    Uses a per-question seed derived from sample_id + topic so that
    the shuffle is reproducible but the answer key distribution is
    uniform across the dataset (not biased by processing order).
    """
    options_list = [correct] + distractors[:3]
    rng = random.Random(f"{sample_id}_{topic}")
    rng.shuffle(options_list)
    answer_key = chr(65 + options_list.index(correct))
    options = {chr(65 + i): opt for i, opt in enumerate(options_list)}
    return options, answer_key


def generate_questions(sample):
    """Generate all possible MCQs from a sample's metadata.

    Each question uses a per-sample RNG seeded from sample_id so that
    distractor selection, question template choice, and option shuffling
    are all reproducible but varied across samples.
    """
    from config import (LAND_USE_CATEGORIES, BUILDING_HEIGHT_CATEGORIES,
                        ROAD_LABELS, URBAN_DENSITY_CATEGORIES,
                        ROAD_SURFACE_BINS, ROAD_SURFACE_OPTIONS,
                        JUNCTION_TYPES, WATER_PROXIMITY_BINS,
                        TRANSIT_DENSITY_BINS)
    questions = []
    sid = sample.get("sample_id", "")
    # Per-sample RNG for distractor and template selection
    rng = random.Random(f"{sid}_qgen")

    # --- LAND USE ---
    lu_cat = sample.get("land_use_category")
    if lu_cat and lu_cat in LAND_USE_CATEGORIES:
        correct = LAND_USE_CATEGORIES[lu_cat]
        distractors = [v for k, v in LAND_USE_CATEGORIES.items() if k != lu_cat]
        rng.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3], sid, "land_use")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["land_use"]),
            "options": opts, "answer": ans,
            "topic": "land_use", "difficulty": "easy",
        })

    # --- BUILDING HEIGHT ---
    levels = sample.get("osm_median_levels")
    if levels and isinstance(levels, (int, float)) and levels > 0:
        levels = int(levels)
        correct = None
        for label, lo, hi in BUILDING_HEIGHT_CATEGORIES:
            if lo <= levels <= hi:
                correct = label
                break
        if correct:
            distractors = [label for label, _, _ in BUILDING_HEIGHT_CATEGORIES
                           if label != correct]
            opts, ans = _shuffle_options(correct, distractors, sid, "building_height")
            questions.append({
                "question": rng.choice(QUESTION_TEMPLATES["building_height"]),
                "options": opts, "answer": ans,
                "topic": "building_height", "difficulty": "easy",
            })

    # --- ROAD TYPE ---
    road_type = sample.get("highway_type")
    if road_type and road_type in ROAD_LABELS:
        correct = ROAD_LABELS[road_type]
        distractors = [v for k, v in ROAD_LABELS.items() if k != road_type]
        rng.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3], sid, "road_type")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["road_type"]),
            "options": opts, "answer": ans,
            "topic": "road_type", "difficulty": "easy",
        })

    # --- URBAN DENSITY ---
    bldg_count = sample.get("osm_building_count")
    if bldg_count is not None and isinstance(bldg_count, (int, float)):
        bldg_count = int(bldg_count)
        correct = None
        for label, lo, hi in URBAN_DENSITY_CATEGORIES:
            if lo <= bldg_count <= hi:
                correct = label
                break
        if correct:
            distractors = [label for label, _, _ in URBAN_DENSITY_CATEGORIES
                           if label != correct]
            opts, ans = _shuffle_options(correct, distractors, sid, "urban_density")
            questions.append({
                "question": rng.choice(QUESTION_TEMPLATES["urban_density"]),
                "options": opts, "answer": ans,
                "topic": "urban_density", "difficulty": "medium",
            })

    # --- PARK / GREEN SPACE ---
    has_park = sample.get("osm_has_park")
    if has_park is True:
        correct = "Yes, green space or parks are present nearby"
        distractors = [
            "No, the area is fully built-up with no visible green space",
            "The area is primarily industrial with no recreational space",
            "Only small private gardens exist, no public green space",
        ]
        opts, ans = _shuffle_options(correct, distractors, sid, "green_space")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["green_space"]),
            "options": opts, "answer": ans,
            "topic": "green_space", "difficulty": "easy",
        })
    elif has_park is False and bldg_count and bldg_count > 20:
        correct = "No, the area is densely built with no parks nearby"
        distractors = [
            "Yes, a large public park is adjacent to this area",
            "Yes, several pocket parks and green corridors exist",
            "The area has extensive waterfront promenades and green space",
        ]
        opts, ans = _shuffle_options(correct, distractors, sid, "green_space")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["green_space"]),
            "options": opts, "answer": ans,
            "topic": "green_space", "difficulty": "easy",
        })

    # --- AMENITY RICHNESS ---
    amenity_count = sample.get("osm_amenity_count")
    if amenity_count is not None and isinstance(amenity_count, (int, float)):
        amenity_count = int(amenity_count)
        if amenity_count >= 20:
            correct = "High — numerous shops, restaurants, and services"
            distractors = [
                "Low — few services, primarily residential",
                "Moderate — some neighborhood shops and cafes",
                "Minimal — industrial area with almost no amenities",
            ]
        elif amenity_count >= 5:
            correct = "Moderate — some neighborhood shops and cafes"
            distractors = [
                "High — numerous shops, restaurants, and services",
                "Low — few services, primarily residential",
                "Minimal — industrial area with almost no amenities",
            ]
        elif amenity_count >= 1:
            correct = "Low — few services, primarily residential"
            distractors = [
                "High — numerous shops, restaurants, and services",
                "Moderate — some neighborhood shops and cafes",
                "Minimal — industrial area with almost no amenities",
            ]
        else:
            correct = "Minimal — industrial area with almost no amenities"
            distractors = [
                "High — numerous shops, restaurants, and services",
                "Moderate — some neighborhood shops and cafes",
                "Low — few services, primarily residential",
            ]
        opts, ans = _shuffle_options(correct, distractors, sid, "amenity_richness")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["amenity_richness"]),
            "options": opts, "answer": ans,
            "topic": "amenity_richness", "difficulty": "medium",
        })

    # --- ROAD SURFACE ---
    surface = sample.get("osm_road_surface")
    if surface and surface in ROAD_SURFACE_BINS:
        correct = ROAD_SURFACE_BINS[surface]
        distractors = [opt for opt in ROAD_SURFACE_OPTIONS if opt != correct]
        rng.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3], sid, "road_surface")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["road_surface"]),
            "options": opts, "answer": ans,
            "topic": "road_surface", "difficulty": "easy",
        })

    # --- JUNCTION TYPE ---
    jtype = sample.get("osm_junction_type")
    if jtype and jtype in JUNCTION_TYPES:
        correct = JUNCTION_TYPES[jtype]
        distractors = [v for k, v in JUNCTION_TYPES.items() if k != jtype]
        rng.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3], sid, "junction_type")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["junction_type"]),
            "options": opts, "answer": ans,
            "topic": "junction_type", "difficulty": "medium",
        })

    # --- WATER PROXIMITY ---
    water_dist = sample.get("osm_water_distance_m")
    if water_dist is not None:
        correct = None
        for label, lo, hi in WATER_PROXIMITY_BINS:
            if lo <= water_dist < hi:
                correct = label
                break
        if correct:
            distractors = [label for label, _, _ in WATER_PROXIMITY_BINS
                           if label != correct]
            if len(distractors) < 3:
                distractors.append("A seasonal water body may be present")
            rng.shuffle(distractors)
            opts, ans = _shuffle_options(correct, distractors[:3], sid, "water_proximity")
            questions.append({
                "question": rng.choice(QUESTION_TEMPLATES["water_proximity"]),
                "options": opts, "answer": ans,
                "topic": "water_proximity", "difficulty": "medium",
            })

    # --- TRANSIT DENSITY ---
    transit_count = sample.get("osm_transit_stop_count")
    if transit_count is not None:
        correct = None
        for label, lo, hi in TRANSIT_DENSITY_BINS:
            if lo <= transit_count <= hi:
                correct = label
                break
        if correct:
            distractors = [label for label, _, _ in TRANSIT_DENSITY_BINS
                           if label != correct]
            rng.shuffle(distractors)
            opts, ans = _shuffle_options(correct, distractors[:3], sid, "transit_density")
            questions.append({
                "question": rng.choice(QUESTION_TEMPLATES["transit_density"]),
                "options": opts, "answer": ans,
                "topic": "transit_density", "difficulty": "medium",
            })

    # --- CAMERA DIRECTION (cross-view) ---
    bearing = sample.get("road_bearing")
    if bearing is not None:
        try:
            bearing = float(bearing)
            correct = bearing_to_quadrant(bearing)
            all_quadrants = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
            distractors = [q for q in all_quadrants if q != correct]
            opts, ans = _shuffle_options(correct, distractors, sid, "camera_direction")
            questions.append({
                "question": rng.choice(QUESTION_TEMPLATES["camera_direction"]),
                "options": opts, "answer": ans,
                "topic": "camera_direction", "difficulty": "medium",
                "sat_marked_path": sample.get("sat_marked_path"),
            })
        except (ValueError, TypeError):
            pass

    # --- MISMATCH BINARY (cross-view, Yes/No) ---
    binary_variants = sample.get("mismatch_binary_variants", [])
    for variant in binary_variants:
        neg_sid = variant.get("negative_sid")
        strategy = variant.get("strategy", "same_city")
        if not neg_sid:
            continue

        difficulty = variant.get("difficulty", "medium")
        sat_marked = sample.get("sat_marked_path")
        own_stv_composite = sample.get("stv_composite_path")
        own_stv_paths = [f"images/sv/{sid}_{a}.jpg"
                         for a in ["along_fwd", "cross_right",
                                   "along_bwd", "cross_left"]]

        # YES variant — matched pair
        yes_opts, yes_ans = _shuffle_options(
            "Yes, the images belong to same location.",
            ["No, images locate to different places."],
            sid, f"mismatch_binary_yes_{strategy}")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["mismatch_binary"]),
            "options": yes_opts, "answer": yes_ans,
            "topic": "mismatch_binary", "difficulty": difficulty,
            "mismatch_strategy": strategy,
            "mismatch_is_match": True,
            "sat_marked_path": sat_marked,
            "stv_shown_paths": own_stv_paths,
            "stv_shown_composite": own_stv_composite,
        })

        # NO variant — mismatched pair
        no_opts, no_ans = _shuffle_options(
            "No, images locate to different places.",
            ["Yes, the images belong to same location."],
            sid, f"mismatch_binary_no_{strategy}")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["mismatch_binary"]),
            "options": no_opts, "answer": no_ans,
            "topic": "mismatch_binary", "difficulty": difficulty,
            "mismatch_strategy": strategy,
            "mismatch_is_match": False,
            "sat_marked_path": sat_marked,
            "mismatch_negative_stv_paths": variant.get("negative_stv_paths"),
            "mismatch_negative_stv_composite": variant.get("negative_stv_composite"),
        })

    # --- MISMATCH MCQ (cross-view, STV grid) ---
    mcq_variants = sample.get("mismatch_mcq_variants", [])
    for variant in mcq_variants:
        composite_path = variant.get("composite_path")
        correct_pos = variant.get("correct_pos")
        strategy = variant.get("strategy", "same_city")
        if not composite_path or not correct_pos:
            continue

        pos_labels = {"A": "Top-Left", "B": "Top-Right",
                      "C": "Bottom-Left", "D": "Bottom-Right"}
        correct = pos_labels[correct_pos]
        distractors = [v for k, v in pos_labels.items() if k != correct_pos]
        opts, ans = _shuffle_options(correct, distractors, sid,
                                     f"mismatch_mcq_{strategy}")
        difficulty = variant.get("difficulty", "hard")
        questions.append({
            "question": rng.choice(QUESTION_TEMPLATES["mismatch_mcq"]),
            "options": opts, "answer": ans,
            "topic": "mismatch_mcq", "difficulty": difficulty,
            "mismatch_strategy": strategy,
            "sat_marked_path": sample.get("sat_marked_path"),
            "composite_stv_path": composite_path,
            "composite_stv_labeled_path": variant.get("composite_labeled_path"),
            "option_stv_paths": variant.get("option_stv_paths"),
            "option_composite_paths": variant.get("option_composite_paths"),
        })

    from config import ENABLED_QUESTION_TYPES
    questions = [q for q in questions if q["topic"] in ENABLED_QUESTION_TYPES]
    return questions


def select_best_question(sample, questions, used_topics_counter):
    """Pick the most distinctive and diverse question for this sample."""
    if not questions:
        return None

    scored = []
    for q in questions:
        topic = q["topic"]
        score = 0

        # Prefer topics not yet used heavily
        usage = used_topics_counter.get(topic, 0)
        score -= usage * 3

        # Prefer visually distinctive answers
        if topic == "building_height":
            levels = sample.get("osm_median_levels", 0) or 0
            if levels > 15 or levels <= 2:
                score += 5  # extremes are more distinctive
            else:
                score += 2
        elif topic == "land_use":
            lu = sample.get("land_use_category", "")
            if lu in ("industrial", "open_space", "institutional"):
                score += 5
            else:
                score += 3
        elif topic == "road_type":
            rt = sample.get("highway_type", "")
            if rt in ("motorway", "trunk", "primary"):
                score += 4
            else:
                score += 2
        elif topic == "urban_density":
            score += 3
        elif topic == "green_space":
            score += 3
        elif topic == "amenity_richness":
            score += 2
        elif topic == "road_surface":
            score += 4
        elif topic == "junction_type":
            jtype = sample.get("osm_junction_type", "")
            score += 5 if jtype == "roundabout" else 3
        elif topic == "water_proximity":
            score += 4
        elif topic == "transit_density":
            score += 2
        elif topic == "camera_direction":
            score += 3
        elif topic == "mismatch_binary":
            score += 3
        elif topic == "mismatch_mcq":
            score += 4

        scored.append((score, q))

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def run(samples=None):
    """Main entry point."""
    print("[Step 5/6] Generating questions...")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "metadata_raw.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    used_topics = Counter()
    total_qs = 0

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        print(f"  [{i+1}/{len(samples)}] {sid}...")

        all_qs = generate_questions(sample)
        total_qs += len(all_qs)
        print(f"    {len(all_qs)} feasible questions")

        # Store ALL feasible questions on the sample
        sample["all_questions"] = all_qs

        # Still pick a "best" for backward compat (flat fields)
        best = select_best_question(sample, all_qs, used_topics)
        if best:
            sample["question"] = best["question"]
            sample["options"] = best["options"]
            sample["answer"] = best["answer"]
            sample["topic"] = best["topic"]
            sample["difficulty"] = best["difficulty"]
            sample["generation_method"] = "template"
            used_topics[best["topic"]] += 1
            print(f"    -> best: [{best['topic']}] {best['answer']}) "
                  f"{best['options'][best['answer']]}")
        else:
            print(f"    WARNING: no questions generated")

    print(f"\n  Total questions generated: {total_qs} across {len(samples)} samples")
    if samples:
        print(f"  Average per sample: {total_qs / len(samples):.1f}")
    print(f"  Best-question topic distribution: {dict(used_topics)}")
    ans_dist = Counter(s.get("answer") for s in samples if s.get("answer"))
    print(f"  Best-question answer distribution: {dict(ans_dist)}")
    return samples


if __name__ == "__main__":
    import pandas as pd
    run()
