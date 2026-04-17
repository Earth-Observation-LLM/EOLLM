"""
Dataset Quality Assessment Script

Reads the merged dataset and produces:
  1. A formatted console report with quality metrics per question type and per city
  2. quality_report.csv — one row per question with all quality flags

Usage:
    python src/assess_quality.py [--input PATH] [--output PATH]

Defaults:
    --input   /home/alperen/Documents/EODATA/dataset_merged.jsonl
    --output  /home/alperen/Documents/EODATA/quality_report.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Config constants (mirrored from src/config.py to avoid import path issues)
# ---------------------------------------------------------------------------

VALID_LAND_USE = {"residential", "commercial", "retail", "industrial", "mixed",
                  "institutional", "open_space", "transport"}

VALID_ROAD_TYPES = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential"}

VALID_ROAD_SURFACES = {
    "asphalt", "paved", "concrete", "concrete:plates",
    "cobblestone", "sett", "paving_stones",
    "unpaved", "gravel", "dirt", "ground", "sand", "compacted",
}

VALID_JUNCTION_TYPES = {"roundabout", "signalized", "unsignalized", "grade_separated"}

MIN_IMAGE_SIZE_BYTES = 5 * 1024  # 5 KB


# ---------------------------------------------------------------------------
# Dimension 1: Metadata completeness
# ---------------------------------------------------------------------------

def check_metadata_complete(topic: str, meta: dict) -> tuple[bool, str]:
    """
    Return (is_complete, reason_if_missing).
    is_complete=True means the OSM field driving this question type is populated
    and has a plausible value.
    """
    if topic == "land_use":
        val = meta.get("land_use_category")
        if not val or val not in VALID_LAND_USE:
            return False, f"land_use_category={val!r} not in valid set"
        return True, ""

    if topic == "building_height":
        val = meta.get("osm_median_building_levels")
        if val is None or val == 0:
            return False, "osm_median_building_levels is null or 0"
        try:
            if float(val) <= 0:
                return False, f"osm_median_building_levels={val} <= 0"
        except (TypeError, ValueError):
            return False, f"osm_median_building_levels={val!r} not numeric"
        return True, ""

    if topic == "road_type":
        val = meta.get("road_type")
        if not val or val not in VALID_ROAD_TYPES:
            return False, f"road_type={val!r} not in valid set"
        return True, ""

    if topic == "urban_density":
        val = meta.get("osm_building_count")
        if val is None:
            return False, "osm_building_count is null"
        return True, ""  # 0 is valid (Low density)

    if topic == "green_space":
        val = meta.get("osm_has_park")
        if val is None:
            return False, "osm_has_park is null"
        return True, ""  # True or False both valid

    if topic == "amenity_richness":
        val = meta.get("osm_amenity_count")
        if val is None:
            return False, "osm_amenity_count is null"
        return True, ""  # 0 is valid (Minimal)

    if topic == "road_surface":
        val = meta.get("osm_road_surface")
        if not val or val not in VALID_ROAD_SURFACES:
            return False, f"osm_road_surface={val!r} not in valid set"
        return True, ""

    if topic == "junction_type":
        val = meta.get("osm_junction_type")
        if not val or val not in VALID_JUNCTION_TYPES:
            return False, f"osm_junction_type={val!r} not in valid set"
        return True, ""

    if topic == "water_proximity":
        val = meta.get("osm_water_distance_m")
        if val is None:
            return False, "osm_water_distance_m is null"
        return True, ""

    if topic == "transit_density":
        val = meta.get("osm_transit_stop_count")
        if val is None:
            return False, "osm_transit_stop_count is null"
        return True, ""  # 0 is valid (None)

    if topic == "camera_direction":
        bearing = meta.get("road_bearing")
        if bearing is None:
            return False, "road_bearing is null"
        try:
            b = float(bearing)
            if not (0 <= b <= 360):
                return False, f"road_bearing={b} out of 0-360 range"
        except (TypeError, ValueError):
            return False, f"road_bearing={bearing!r} not numeric"
        arrow_paths = meta.get("camera_arrow_paths", {})
        if len(arrow_paths) < 4:
            return False, f"camera_arrow_paths has only {len(arrow_paths)}/4 entries"
        return True, ""

    if topic == "mismatch_binary":
        variants = meta.get("mismatch_binary_variants", [])
        if not variants:
            return False, "mismatch_binary_variants is empty"
        return True, ""

    if topic == "mismatch_mcq":
        variants = meta.get("mismatch_mcq_variants", [])
        if not variants:
            return False, "mismatch_mcq_variants is empty"
        for v in variants:
            if not v.get("composite_path") or not v.get("correct_pos"):
                return False, "mismatch_mcq variant missing composite_path or correct_pos"
        return True, ""

    # Unknown topic
    return True, ""


# ---------------------------------------------------------------------------
# Dimension 2: Cross-signal plausibility
# ---------------------------------------------------------------------------

# Cities where skyscrapers (20+ floors) are contextually plausible
SKYSCRAPER_CITIES = {
    "New York City", "Tokyo", "Singapore", "Chicago", "Seoul",
    "Hong Kong", "Shanghai", "Dubai", "Toronto", "Sydney",
    "London", "Paris", "Moscow", "St. Petersburg", "Istanbul",
    "Mumbai", "Buenos Aires", "São Paulo", "Rio de Janeiro",
    "Bangkok", "Kuala Lumpur", "Taipei", "Mexico City",
}


def check_plausibility(topic: str, meta: dict, city: str) -> list[str]:
    """
    Return a list of suspicion strings. Empty list = plausible.
    These are heuristic flags — not definitive errors.
    """
    flags = []
    lu = meta.get("land_use_category", "")
    bc = meta.get("osm_building_count") or 0
    ac = meta.get("osm_amenity_count") or 0
    has_park = meta.get("osm_has_park")
    levels = meta.get("osm_median_building_levels") or 0
    road_type = meta.get("road_type", "")
    junction = meta.get("osm_junction_type", "")

    if topic == "land_use":
        if lu == "open_space" and bc > 50:
            flags.append("open_space but building_count>50")
        if lu == "residential" and bc == 0:
            flags.append("residential but building_count=0")
        if lu == "industrial" and ac > 15:
            flags.append("industrial but amenity_count>15")

    if topic == "urban_density":
        # "Very high density" requires building_count > 150 by definition,
        # but flag if amenity count is also 0 — unusual for dense urban cores
        if bc > 150 and ac == 0:
            flags.append("very_high_density but amenity_count=0")

    if topic == "green_space":
        if has_park and lu == "industrial" and bc > 80:
            flags.append("has_park=True but heavy industrial area (lu=industrial, bc>80)")

    if topic == "building_height":
        try:
            lv = int(float(levels))
        except (TypeError, ValueError):
            lv = 0
        if lv >= 21 and lu == "residential" and city not in SKYSCRAPER_CITIES:
            flags.append(f"skyscraper height but lu=residential in non-skyscraper city ({city})")
        if lv > 100:
            flags.append(f"implausible building levels: {lv}")

    if topic == "road_type":
        if road_type == "motorway" and junction == "unsignalized":
            flags.append("motorway but junction_type=unsignalized")

    return flags


# ---------------------------------------------------------------------------
# Dimension 3: Image completeness (image-dependent types only)
# ---------------------------------------------------------------------------

IMAGE_DEPENDENT_TYPES = {"camera_direction", "mismatch_binary", "mismatch_mcq"}

# Base directories where images might be found
# The merged dataset has samples from multiple runs; image paths are relative to each run's output/
# We store the run base directories to try resolving paths
RUN_BASE_DIRS = [
    "/home/alperen/Documents/EODATA/output_1445/output",
    "/home/alperen/Documents/EODATA/output_673img/output",
    "/home/alperen/Documents/EODATA/output_707/output",
    "/home/alperen/Documents/EODATA/output_1459_28_03/output",
    "/home/alperen/Documents/EOLLM/output",
]


def _image_ok(rel_path: str) -> bool:
    """Check if an image file exists and is > MIN_IMAGE_SIZE_BYTES."""
    if not rel_path:
        return False
    for base in RUN_BASE_DIRS:
        full = os.path.join(base, rel_path)
        if os.path.isfile(full):
            return os.path.getsize(full) >= MIN_IMAGE_SIZE_BYTES
    return False


def check_images(topic: str, question: dict, meta: dict) -> tuple[bool, str]:
    """
    For image-dependent question types, verify all referenced images are present.
    Returns (ok, reason_if_bad). Non-image types always return (True, "").
    """
    if topic not in IMAGE_DEPENDENT_TYPES:
        return True, ""

    if topic == "camera_direction":
        arrow_paths = meta.get("camera_arrow_paths", {})
        missing = [k for k, p in arrow_paths.items() if not _image_ok(p)]
        if missing:
            return False, f"missing/small arrow images: {missing}"
        return True, ""

    if topic == "mismatch_binary":
        variants = meta.get("mismatch_binary_variants", [])
        for v in variants:
            composite = v.get("negative_stv_composite") or v.get("composite_stv_path")
            if composite and not _image_ok(composite):
                return False, f"missing mismatch_binary composite: {composite}"
        return True, ""

    if topic == "mismatch_mcq":
        variants = meta.get("mismatch_mcq_variants", [])
        for v in variants:
            composite = v.get("composite_path")
            if composite and not _image_ok(composite):
                return False, f"missing mismatch_mcq composite: {composite}"
        return True, ""

    return True, ""


# ---------------------------------------------------------------------------
# Per-question assessment
# ---------------------------------------------------------------------------

def assess_question(question: dict, meta: dict, city: str, sample_id: str) -> dict:
    """
    Assess a single question and return a quality record dict.
    """
    topic = question.get("topic", "unknown")
    difficulty = question.get("difficulty", "")
    answer = question.get("answer", "")
    options = question.get("options", {})

    # Dimension 1
    meta_ok, meta_reason = check_metadata_complete(topic, meta)

    # Dimension 2
    plausibility_flags = check_plausibility(topic, meta, city)

    # Dimension 3
    img_ok, img_reason = check_images(topic, question, meta)
    images_ok_val = "" if topic not in IMAGE_DEPENDENT_TYPES else (1 if img_ok else 0)

    # Structural checks
    struct_flags = []
    if answer not in options:
        struct_flags.append("answer_not_in_options")
    option_vals = list(options.values())
    if len(option_vals) != len(set(option_vals)):
        struct_flags.append("duplicate_options")

    # Overall quality label
    if struct_flags:
        overall = "invalid"
    elif not meta_ok:
        overall = "low_metadata"
    elif not img_ok and topic in IMAGE_DEPENDENT_TYPES:
        overall = "missing_images"
    elif plausibility_flags:
        overall = "suspicious"
    else:
        overall = "pass"

    all_reasons = []
    if meta_reason:
        all_reasons.append(meta_reason)
    all_reasons.extend(plausibility_flags)
    if img_reason:
        all_reasons.append(img_reason)
    all_reasons.extend(struct_flags)

    return {
        "sample_id": sample_id,
        "city": city,
        "topic": topic,
        "difficulty": difficulty,
        "answer": answer,
        "metadata_complete": 1 if meta_ok else 0,
        "plausibility_flag": 1 if plausibility_flags else 0,
        "images_ok": images_ok_val,
        "overall_quality": overall,
        "flag_reasons": "; ".join(all_reasons),
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _pct(num, den):
    return 100 * num / den if den else 0.0


def print_report(all_results: list[dict], samples_meta: dict):
    """Print formatted quality report to stdout."""
    W = 80

    print("=" * W)
    print("EOLLM DATASET QUALITY REPORT")
    print("=" * W)

    total_samples = len(samples_meta)
    total_questions = len(all_results)
    sv_missing = sum(1 for s in samples_meta.values() if s.get("sv_count", 4) < 4)

    print(f"\nTotal samples    : {total_samples:,}")
    print(f"Total questions  : {total_questions:,}")
    print(f"Avg per sample   : {total_questions / total_samples:.1f}")
    print(f"Samples <4 SV    : {sv_missing:,}  ({_pct(sv_missing, total_samples):.1f}%)")

    # ---- Section 1: Questions by type ----
    print(f"\n{'─'*W}")
    print("SECTION 1: Questions by Type")
    print(f"{'─'*W}")
    print(f"  {'Type':<25} {'Count':>7}  {'%Total':>7}  {'Meta OK':>8}  {'Plaus OK':>9}  {'Overall Pass':>13}")

    by_topic = defaultdict(list)
    for r in all_results:
        by_topic[r["topic"]].append(r)

    for topic in sorted(by_topic, key=lambda t: -len(by_topic[t])):
        rows = by_topic[topic]
        n = len(rows)
        pct_total = _pct(n, total_questions)
        meta_ok = sum(r["metadata_complete"] for r in rows)
        plaus_ok = sum(1 for r in rows if r["plausibility_flag"] == 0)
        passing = sum(1 for r in rows if r["overall_quality"] == "pass")
        # suffix for recommendations
        suffix = ""
        if n < 1000:
            suffix = "  ← SPARSE"
        elif _pct(meta_ok, n) < 60:
            suffix = "  ← LOW METADATA"
        print(f"  {topic:<25} {n:>7,}  {pct_total:>6.1f}%  {_pct(meta_ok,n):>7.1f}%  {_pct(plaus_ok,n):>8.1f}%  {_pct(passing,n):>12.1f}%{suffix}")

    # ---- Section 2: City x Type metadata completeness matrix ----
    print(f"\n{'─'*W}")
    print("SECTION 2: Metadata Completeness (% OK) by City × Question Type")
    print(f"  (values < 80% are flagged with *)")
    print(f"{'─'*W}")

    all_cities = sorted(set(r["city"] for r in all_results))
    all_topics = sorted(by_topic.keys())
    # column topics to show (skip water_proximity for space)
    col_topics = [t for t in all_topics if t != "water_proximity"]

    # Build lookup: city → topic → [records]
    city_topic = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        city_topic[r["city"]][r["topic"]].append(r)

    # Print header
    header = f"  {'City':<20}"
    for t in col_topics:
        abbrev = t[:7]
        header += f" {abbrev:>7}"
    print(header)
    print("  " + "-" * (20 + 8 * len(col_topics)))

    for city in all_cities:
        row = f"  {city:<20}"
        for t in col_topics:
            rows = city_topic[city][t]
            if not rows:
                row += f"  {'N/A':>5}"
            else:
                pct = _pct(sum(r["metadata_complete"] for r in rows), len(rows))
                marker = "*" if pct < 80 else " "
                row += f" {pct:>6.0f}{marker}"
        print(row)

    # water_proximity separate note
    wp_rows = by_topic.get("water_proximity", [])
    if wp_rows:
        wp_meta_ok = _pct(sum(r["metadata_complete"] for r in wp_rows), len(wp_rows))
        print(f"\n  water_proximity: {len(wp_rows):,} questions, {wp_meta_ok:.1f}% metadata OK  ← RECOMMEND REMOVAL")

    # ---- Section 3: Plausibility flags ----
    print(f"\n{'─'*W}")
    print("SECTION 3: Cross-Signal Plausibility Flags")
    print(f"{'─'*W}")

    for topic in sorted(by_topic):
        rows = by_topic[topic]
        flagged = [r for r in rows if r["plausibility_flag"] == 1]
        if not flagged:
            continue
        print(f"\n  {topic} — {len(flagged):,}/{len(rows):,} flagged ({_pct(len(flagged),len(rows)):.1f}%)")
        reason_counter = Counter()
        for r in flagged:
            for reason in r["flag_reasons"].split("; "):
                if reason:
                    reason_counter[reason] += 1
        for reason, cnt in reason_counter.most_common(5):
            print(f"    {cnt:>5}x  {reason}")

    # ---- Section 4: Image completeness ----
    print(f"\n{'─'*W}")
    print("SECTION 4: Image Completeness (image-based question types)")
    print(f"{'─'*W}")
    for topic in ["camera_direction", "mismatch_binary", "mismatch_mcq"]:
        rows = by_topic.get(topic, [])
        if not rows:
            continue
        img_rows = [r for r in rows if r["images_ok"] != ""]
        if not img_rows:
            print(f"  {topic:<25} — image check skipped (no image path data)")
            continue
        ok = sum(1 for r in img_rows if r["images_ok"] == 1)
        print(f"  {topic:<25} {ok:,}/{len(img_rows):,} images OK  ({_pct(ok, len(img_rows)):.1f}%)")

    # ---- Section 5: Answer distribution ----
    print(f"\n{'─'*W}")
    print("SECTION 5: Answer Distribution per Question Type")
    print(f"  (well-balanced MCQ should be ~25% per letter; binary ~50%)")
    print(f"{'─'*W}")
    for topic in sorted(by_topic):
        rows = by_topic[topic]
        dist = Counter(r["answer"] for r in rows)
        total_t = len(rows)
        pcts = {k: _pct(v, total_t) for k, v in dist.items()}
        max_pct = max(pcts.values()) if pcts else 0
        skew_marker = "  ← SKEWED" if max_pct > 40 else ""
        line = f"  {topic:<25} " + "  ".join(f"{k}:{v:.0f}%" for k, v in sorted(pcts.items()))
        print(line + skew_marker)

    # ---- Section 6: Per-city summary ----
    print(f"\n{'─'*W}")
    print("SECTION 6: Per-City Summary")
    print(f"{'─'*W}")
    print(f"  {'City':<25} {'Samples':>8}  {'Questions':>10}  {'Avg Q':>6}  {'Pass%':>7}  {'SV Full':>8}")

    for city in sorted(all_cities):
        city_samples = {sid: d for sid, d in samples_meta.items() if d["city"] == city}
        n_samples = len(city_samples)
        city_rows = [r for r in all_results if r["city"] == city]
        n_q = len(city_rows)
        avg_q = n_q / n_samples if n_samples else 0
        passing = _pct(sum(1 for r in city_rows if r["overall_quality"] == "pass"), n_q)
        sv_full = sum(1 for s in city_samples.values() if s.get("sv_count", 4) >= 4)
        sv_pct = _pct(sv_full, n_samples)
        print(f"  {city:<25} {n_samples:>8,}  {n_q:>10,}  {avg_q:>6.1f}  {passing:>6.1f}%  {sv_pct:>7.1f}%")

    # ---- Section 7: Recommendations ----
    print(f"\n{'─'*W}")
    print("SECTION 7: FILTERING RECOMMENDATIONS")
    print(f"{'─'*W}")

    recs = []

    # water_proximity
    wp = by_topic.get("water_proximity", [])
    if wp:
        recs.append(("[REMOVE TYPE]   ", "water_proximity",
                     f"only {len(wp):,} questions ({_pct(len(wp),total_questions):.1f}% of total)"))

    # Low metadata coverage
    for topic, rows in by_topic.items():
        meta_pct = _pct(sum(r["metadata_complete"] for r in rows), len(rows))
        if meta_pct < 60 and topic != "water_proximity":
            recs.append(("[LOW METADATA]  ", topic, f"{meta_pct:.1f}% metadata completeness"))

    # High plausibility flag rate
    for topic, rows in by_topic.items():
        flag_pct = _pct(sum(r["plausibility_flag"] for r in rows), len(rows))
        if flag_pct > 5:
            recs.append(("[REVIEW]        ", topic, f"{flag_pct:.1f}% plausibility flags"))

    # Cities with low avg question count (pipeline issues)
    for city in all_cities:
        city_rows = [r for r in all_results if r["city"] == city]
        city_samples = sum(1 for s in samples_meta.values() if s["city"] == city)
        avg_q = len(city_rows) / city_samples if city_samples else 0
        if avg_q < 12:
            recs.append(("[CHECK CITY]    ", city, f"avg {avg_q:.1f} questions/sample (low)"))

    # Image missing issues
    for topic in ["camera_direction", "mismatch_binary", "mismatch_mcq"]:
        rows = by_topic.get(topic, [])
        img_rows = [r for r in rows if r["images_ok"] != ""]
        if img_rows:
            missing_pct = _pct(sum(1 for r in img_rows if r["images_ok"] == 0), len(img_rows))
            if missing_pct > 5:
                recs.append(("[MISSING IMAGES]", topic, f"{missing_pct:.1f}% of questions have missing images"))

    if not recs:
        print("  No significant issues found.")
    else:
        for tag, name, reason in recs:
            print(f"  {tag}  {name:<25}  — {reason}")

    print(f"\n{'='*W}")
    print(f"Report complete. See quality_report.csv for per-question details.")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sample_id", "city", "topic", "difficulty", "answer",
    "metadata_complete", "plausibility_flag", "images_ok",
    "overall_quality", "flag_reasons",
]


def write_csv(all_results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV written: {output_path}  ({len(all_results):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assess quality of merged EOLLM dataset")
    parser.add_argument(
        "--input",
        default="/home/alperen/Documents/EODATA/dataset_merged.jsonl",
        help="Path to dataset_merged.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/alperen/Documents/EODATA/quality_report.csv",
        help="Path for output quality_report.csv",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {args.input} ...")
    all_results = []
    samples_meta = {}  # sample_id -> {city, sv_count}

    with open(args.input, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {lineno}: JSON parse error — {e}", file=sys.stderr)
                continue

            sample_id = rec.get("sample_id", f"unknown_{lineno}")
            city = rec.get("location", {}).get("city", "Unknown")
            meta = rec.get("metadata", {})
            sv_count = rec.get("validation", {}).get("streetview_count", 4)

            samples_meta[sample_id] = {"city": city, "sv_count": sv_count}

            for question in rec.get("questions", []):
                result = assess_question(question, meta, city, sample_id)
                all_results.append(result)

    print(f"Loaded {len(samples_meta):,} samples, {len(all_results):,} questions.\n")

    print_report(all_results, samples_meta)
    write_csv(all_results, args.output)


if __name__ == "__main__":
    main()
