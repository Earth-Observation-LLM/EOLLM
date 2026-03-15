"""
Urban VQA Dataset Pipeline — City-Agnostic Orchestrator
========================================================
Generates VQA samples across multiple global cities.

Usage:
    python src/run_pipeline.py                    # All 6 cities, 10 samples each
    python src/run_pipeline.py --cities nyc paris  # Specific cities
    python src/run_pipeline.py --samples 5         # 5 samples per city
"""

import os
import sys
import json
import time
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def build_jsonl_record(sample):
    """Convert enriched sample dict into final JSONL record."""
    from config import LAND_USE_CATEGORIES
    sid = sample["sample_id"]

    images = {
        "satellite": f"images/sat/{sid}.png",
        "satellite_source": sample.get("satellite_actual_source",
                                       sample.get("satellite_source", "unknown")),
        "satellite_date": sample.get("satellite_date", "unknown"),
        "streetview_along_fwd": f"images/sv/{sid}_along_fwd.jpg",
        "streetview_along_bwd": f"images/sv/{sid}_along_bwd.jpg",
        "streetview_cross_left": f"images/sv/{sid}_cross_left.jpg",
        "streetview_cross_right": f"images/sv/{sid}_cross_right.jpg",
        "streetview_source": "Google Street View",
        "streetview_road_bearing": sample.get("road_bearing"),
        "streetview_date": sample.get("sv_date") if isinstance(sample.get("sv_date"), str) else None,
    }

    if sample.get("composite_4sat_path"):
        images["composite_4sat"] = sample["composite_4sat_path"]

    options = sample.get("options", {})
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            try:
                import ast
                options = ast.literal_eval(options)
            except Exception:
                options = {}

    # Parse list/dict fields that may be serialized as strings
    amenity_types = sample.get("osm_amenity_types", [])
    if isinstance(amenity_types, str):
        try:
            amenity_types = json.loads(amenity_types)
        except Exception:
            amenity_types = []

    building_types = sample.get("osm_building_types", {})
    if isinstance(building_types, str):
        try:
            building_types = json.loads(building_types)
        except Exception:
            building_types = {}

    lu_cat = sample.get("land_use_category", "unknown")
    lu_label = LAND_USE_CATEGORIES.get(lu_cat, lu_cat)

    metadata = {
        "land_use_category": lu_cat,
        "land_use_label": lu_label,
        "metadata_radius_m": sample.get("metadata_radius_m", 200),
        "osm_building_count": _safe_int(sample.get("osm_building_count")),
        "osm_median_building_levels": _safe_int(sample.get("osm_median_levels")),
        "osm_dominant_landuse_raw": sample.get("osm_dominant_landuse_raw"),
        "osm_amenity_count": _safe_int(sample.get("osm_amenity_count")),
        "osm_amenity_types": amenity_types,
        "osm_has_park": bool(sample.get("osm_has_park")),
        "osm_building_types": building_types,
        "road_name": sample.get("road_name"),
        "road_type": sample.get("highway_type"),
        "osm_road_surface": sample.get("osm_road_surface"),
        "osm_junction_type": sample.get("osm_junction_type"),
        "osm_water_distance_m": sample.get("osm_water_distance_m"),
        "osm_transit_stop_count": _safe_int(sample.get("osm_transit_stop_count")),
    }

    # Add US Census data if available
    if sample.get("census_tract"):
        metadata["census_tract"] = sample["census_tract"]
        metadata["census_population"] = _safe_int(sample.get("census_population"))
        metadata["census_median_income"] = _safe_int(sample.get("census_median_income"))

    # Validation
    validation_issues = sample.get("validation_issues", [])
    if isinstance(validation_issues, str):
        try:
            validation_issues = json.loads(validation_issues)
        except Exception:
            validation_issues = []

    sv_labels = ["along_fwd", "along_bwd", "cross_left", "cross_right"]
    sv_count = sum(
        1 for l in sv_labels
        if os.path.exists(os.path.join(ROOT, "output", "images", "sv", f"{sid}_{l}.jpg"))
    )
    sat_present = os.path.exists(
        os.path.join(ROOT, "output", "images", "sat", f"{sid}.png")
    )

    return {
        "sample_id": sid,
        "location": {
            "latitude": sample["lat"],
            "longitude": sample["lon"],
            "city": sample.get("city_name", ""),
            "country": sample.get("country", ""),
            "neighborhood": sample.get("neighborhood", ""),
            "road_name": sample.get("road_name", ""),
            "geo_suburb": sample.get("geo_suburb", ""),
        },
        "images": images,
        "question": sample.get("question", ""),
        "options": options,
        "answer": sample.get("answer", ""),
        "difficulty": sample.get("difficulty", ""),
        "topic": sample.get("topic", ""),
        "generation_method": "template",
        "metadata": metadata,
        "validation": {
            "all_images_present": sat_present and sv_count == 4,
            "satellite_present": sat_present,
            "streetview_count": sv_count,
            "issues": validation_issues,
        },
    }


def _safe_int(val):
    if val is None:
        return None
    try:
        v = int(float(val))
        return v if v >= 0 else None
    except (ValueError, TypeError):
        return None


def print_summary(records):
    """Print a summary of the generated dataset."""
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total samples: {len(records)}")

    valid = sum(1 for r in records if not any(
        "CRITICAL" in i for i in r["validation"]["issues"]
    ))
    print(f"  Valid samples: {valid}/{len(records)}")

    # City distribution
    cities = Counter(r["location"]["city"] for r in records)
    print(f"\n  Cities ({len(cities)}):")
    for city, count in cities.most_common():
        print(f"    {city}: {count}")

    # Topic distribution
    topics = Counter(r["topic"] for r in records if r["topic"])
    print(f"\n  Topics ({len(topics)}):")
    for topic, count in topics.most_common():
        print(f"    {topic}: {count}")

    # Answer distribution
    answers = Counter(r["answer"] for r in records if r["answer"])
    print(f"\n  Answer distribution:")
    for ans in sorted(answers):
        print(f"    {ans}: {answers[ans]}")

    # Land use diversity
    lus = Counter(r["metadata"]["land_use_label"] for r in records
                  if r["metadata"]["land_use_label"])
    print(f"\n  Land use ({len(lus)} types):")
    for lu, count in lus.most_common():
        print(f"    {lu}: {count}")

    # Images
    sat_ok = sum(1 for r in records if r["validation"]["satellite_present"])
    sv_full = sum(1 for r in records if r["validation"]["streetview_count"] == 4)
    print(f"\n  Images:")
    print(f"    Satellite: {sat_ok}/{len(records)}")
    print(f"    Street View (4/4): {sv_full}/{len(records)}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Urban VQA Pipeline")
    parser.add_argument("--cities", nargs="+", help="City keys to process")
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples per city (default: 10)")
    args = parser.parse_args()

    from config import CITIES

    if args.cities:
        cities = {k: CITIES[k] for k in args.cities if k in CITIES}
        unknown = [k for k in args.cities if k not in CITIES]
        if unknown:
            print(f"Unknown cities: {unknown}. Available: {list(CITIES.keys())}")
    else:
        cities = CITIES

    start_time = time.time()
    total = len(cities) * args.samples
    print("=" * 65)
    print(f"  Urban VQA Dataset Pipeline")
    print(f"  {len(cities)} cities x {args.samples} samples = {total} total")
    print(f"  Cities: {', '.join(c['name'] for c in cities.values())}")
    print("=" * 65)

    os.makedirs(os.path.join(ROOT, "output", "images", "sat"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output", "images", "sv"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output", "images", "composite"), exist_ok=True)

    from importlib import import_module
    step1 = import_module("01_sample_locations")
    step2 = import_module("02_fetch_satellite")
    step3 = import_module("03_fetch_streetview")
    step4 = import_module("04_enrich_metadata")
    step_composite = import_module("07_generate_composites")
    step5 = import_module("05_generate_questions")
    step6 = import_module("06_validate")

    samples = step1.run(cities=cities, num_samples=args.samples)
    samples = step2.run(samples)
    samples = step3.run(samples)
    samples = step4.run(samples)
    samples = step_composite.run(samples)
    samples = step5.run(samples)
    samples = step6.run(samples)

    # Build JSONL
    print("\n  Building dataset.jsonl...")
    jsonl_path = os.path.join(ROOT, "output", "dataset.jsonl")
    records = []
    with open(jsonl_path, 'w') as f:
        for sample in samples:
            record = build_jsonl_record(sample)
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {jsonl_path}")

    print_summary(records)
    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
