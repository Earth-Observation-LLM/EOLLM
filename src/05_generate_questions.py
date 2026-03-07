"""
Step 5: Generate template-based MCQs from OSM metadata.
City-agnostic — all questions derived from universal OSM data.
"""

import os
import json
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shuffle_options(correct, distractors):
    """Shuffle options and return (options_dict, answer_key)."""
    options_list = [correct] + distractors[:3]
    random.shuffle(options_list)
    answer_key = chr(65 + options_list.index(correct))
    options = {chr(65 + i): opt for i, opt in enumerate(options_list)}
    return options, answer_key


def generate_questions(sample):
    """Generate all possible MCQs from a sample's metadata."""
    from config import (LAND_USE_CATEGORIES, BUILDING_HEIGHT_CATEGORIES,
                        ROAD_LABELS, URBAN_DENSITY_CATEGORIES)
    questions = []

    # --- LAND USE ---
    lu_cat = sample.get("land_use_category")
    if lu_cat and lu_cat in LAND_USE_CATEGORIES:
        correct = LAND_USE_CATEGORIES[lu_cat]
        distractors = [v for k, v in LAND_USE_CATEGORIES.items() if k != lu_cat]
        # Prefer plausible distractors
        random.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3])
        questions.append({
            "question": "What is the primary land use character of this area?",
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
            opts, ans = _shuffle_options(correct, distractors)
            questions.append({
                "question": "What building height category best describes the dominant structures in this area?",
                "options": opts, "answer": ans,
                "topic": "building_height", "difficulty": "easy",
            })

    # --- ROAD TYPE ---
    road_type = sample.get("highway_type")
    if road_type and road_type in ROAD_LABELS:
        correct = ROAD_LABELS[road_type]
        distractors = [v for k, v in ROAD_LABELS.items() if k != road_type]
        random.shuffle(distractors)
        opts, ans = _shuffle_options(correct, distractors[:3])
        questions.append({
            "question": "What type of road infrastructure is dominant at this location?",
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
            opts, ans = _shuffle_options(correct, distractors)
            questions.append({
                "question": "What is the approximate urban density level of this area based on building concentration?",
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
        opts, ans = _shuffle_options(correct, distractors)
        questions.append({
            "question": "Is there publicly accessible green space or parkland near this location?",
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
        opts, ans = _shuffle_options(correct, distractors)
        questions.append({
            "question": "Is there publicly accessible green space or parkland near this location?",
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
        opts, ans = _shuffle_options(correct, distractors)
        questions.append({
            "question": "What is the level of commercial amenity presence in this area?",
            "options": opts, "answer": ans,
            "topic": "amenity_richness", "difficulty": "medium",
        })

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

        scored.append((score, q))

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def run(samples=None):
    """Main entry point."""
    print("[Step 5/6] Generating questions...")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "metadata_raw.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    random.seed(42)
    used_topics = Counter()

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        print(f"  [{i+1}/{len(samples)}] {sid}...")

        all_qs = generate_questions(sample)
        print(f"    {len(all_qs)} candidate questions")

        best = select_best_question(sample, all_qs, used_topics)
        if best:
            sample["question"] = best["question"]
            sample["options"] = best["options"]
            sample["answer"] = best["answer"]
            sample["topic"] = best["topic"]
            sample["difficulty"] = best["difficulty"]
            sample["generation_method"] = "template"
            used_topics[best["topic"]] += 1
            print(f"    -> [{best['topic']}] {best['answer']}) "
                  f"{best['options'][best['answer']]}")
        else:
            print(f"    WARNING: no questions generated")

    print(f"\n  Topic distribution: {dict(used_topics)}")
    ans_dist = Counter(s.get("answer") for s in samples if s.get("answer"))
    print(f"  Answer distribution: {dict(ans_dist)}")
    return samples


if __name__ == "__main__":
    import pandas as pd
    run()
