"""Flatten hierarchical location records into individual question records."""

import os
import sys

# Allow imports from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CITIES


# Build city_key -> canonical name mapping from config
CITY_KEY_TO_NAME = {key: info["name"] for key, info in CITIES.items()}


def extract_city_key(sample_id: str, known_keys=None) -> str:
    """Extract city key from sample_id by matching against known city keys.

    E.g., 'st_petersburg_0042' -> 'st_petersburg'
    Tries longest match first to handle multi-underscore keys.
    """
    if known_keys is None:
        known_keys = CITY_KEY_TO_NAME.keys()
    for key in sorted(known_keys, key=len, reverse=True):
        if sample_id.startswith(key + "_"):
            remainder = sample_id[len(key) + 1:]
            if remainder.isdigit():
                return key
    return None


def sid_to_city_name(sample_id: str, known_keys=None) -> str:
    """Convert sample_id to canonical city name.

    Returns the canonical name from CITIES config, or None if not found.
    """
    city_key = extract_city_key(sample_id, known_keys)
    if city_key and city_key in CITY_KEY_TO_NAME:
        return CITY_KEY_TO_NAME[city_key]
    return None


def extract_sid_from_stv_path(path: str) -> str:
    """Extract sample_id from a streetview image path.

    'images/sv/moscow_0032_along_fwd.jpg' -> 'moscow_0032'
    """
    filename = os.path.basename(path)
    # Remove extension
    name = filename.rsplit(".", 1)[0]
    # Remove angle suffix
    for suffix in ["_along_fwd", "_along_bwd", "_cross_left", "_cross_right"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def flatten_record(record: dict) -> list[dict]:
    """Flatten a single location record into individual question records.

    Each question in the record's 'questions' array becomes its own record
    with parent location metadata attached.
    """
    sample_id = record["sample_id"]
    location = record.get("location", {})
    metadata = record.get("metadata", {})
    validation = record.get("validation", {})
    images = record.get("images", {})

    city = location.get("city", "")
    country = location.get("country", "")
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    land_use = metadata.get("land_use_category", "")
    streetview_count = validation.get("streetview_count", 0)

    # Track topic occurrence index for unique question_id generation
    topic_counts = {}
    flat_records = []

    for q in record.get("questions", []):
        topic = q.get("topic", "unknown")
        idx = topic_counts.get(topic, 0)
        topic_counts[topic] = idx + 1

        question_id = f"{sample_id}_{topic}_{idx}"

        flat = {
            # Identity
            "question_id": question_id,
            "sample_id": sample_id,
            # Location metadata
            "city": city,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            "land_use": land_use,
            # Validation
            "streetview_count": streetview_count,
            # Question fields
            "question": q.get("question", ""),
            "options": q.get("options", {}),
            "answer": q.get("answer", ""),
            "topic": topic,
            "difficulty": q.get("difficulty", ""),
            "generation_method": q.get("generation_method", ""),
            # Image paths - common
            "images": images,
            "sat_marked_path": q.get("sat_marked_path"),
            # Camera direction specific
            "query_stv_path": q.get("query_stv_path"),
            "query_stv_angle": q.get("query_stv_angle"),
            "option_sat_paths": q.get("option_sat_paths"),
            # Mismatch specific
            "mismatch_strategy": q.get("mismatch_strategy"),
            "mismatch_is_match": q.get("mismatch_is_match"),
            "mismatch_negative_stv_paths": q.get("mismatch_negative_stv_paths"),
            "mismatch_negative_stv_composite": q.get(
                "mismatch_negative_stv_composite"
            ),
            "composite_stv_path": q.get("composite_stv_path"),
            "composite_stv_labeled_path": q.get("composite_stv_labeled_path"),
            "option_stv_paths": q.get("option_stv_paths"),
            "option_composite_paths": q.get("option_composite_paths"),
            "stv_shown_paths": q.get("stv_shown_paths"),
            "stv_shown_composite": q.get("stv_shown_composite"),
        }

        flat_records.append(flat)

    return flat_records


def flatten_dataset(records: list[dict]) -> list[dict]:
    """Flatten all location records into question-level records.

    Args:
        records: List of hierarchical location records (from dataset JSONL).

    Returns:
        List of flat question records.
    """
    flat = []
    for rec in records:
        flat.extend(flatten_record(rec))

    print(f"  Flattened {len(records)} locations -> {len(flat)} questions")
    return flat
