import re
import json
from pathlib import Path


SEEN_CITIES = {
    "Amsterdam", "Ankara", "Antalya", "Athens", "Barcelona", "Berlin",
    "Brussels", "Budapest", "Buenos Aires", "Bursa", "Helsinki",
    "Izmir", "Kayseri", "Lisbon", "London", "Mexico City", "Mumbai",
    "Nairobi", "New York City", "Nicosia", "Paris", "Reykjavik",
    "Rome", "Samsun", "St. Petersburg", "Stockholm", "Taipei",
    "Tallinn", "Tokyo", "Toronto", "Vienna", "Zurich"
}

UNSEEN_CITIES = {
    "Cape Town", "Chicago", "Istanbul", "Moscow", "Rio de Janeiro",
    "Seoul", "Singapore", "Sydney"
}


def load_jsonl(path):
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_answer(text, options=None):
    """Extract a single answer letter from model output.

    Tries in order:
      1. Exact single-letter match (A/B/C/D)
      2. First bare letter found via word-boundary regex
      3. Match against the option value strings (handles models that echo the
         full option text instead of just the letter)

    ``options`` should be the dict from the record e.g. {"A": "Yes", "B": "No"}.
    It is optional — without it steps 1 and 2 still run.
    """
    if text is None:
        return None

    text_upper = str(text).strip().upper()

    # 1. Exact letter
    if text_upper in {"A", "B", "C", "D"}:
        return text_upper

    # 2. First bare letter in the string
    match = re.search(r"\b([ABCD])\b", text_upper)
    if match:
        return match.group(1)

    # 3. Model echoed the full option value — match against options dict
    if options:
        for letter, value in options.items():
            if letter not in {"A", "B", "C", "D"}:
                continue
            value_upper = str(value).strip().upper()
            if value_upper and (value_upper in text_upper or text_upper in value_upper):
                return letter

    return None


def get_city_type(city):
    if city in SEEN_CITIES:
        return "seen"
    if city in UNSEEN_CITIES:
        return "unseen"
    return "unknown"


def select_images_for_record(record):
    topic = record.get("topic", "")
    img = record.get("images", {})
    selected = []

    def add(label, path):
        if isinstance(path, str) and path.startswith("images/"):
            selected.append({"label": label, "path": path})

    if topic == "camera_direction":
        add("Query street-view image", record.get("query_stv_path"))

        for opt, path in sorted((record.get("option_sat_paths") or {}).items()):
            add(f"Option {opt} satellite arrow image", path)

        return selected

    if topic.startswith("mismatch_binary"):
        add("Marked satellite image", record.get("sat_marked_path"))
        # is_match=True:  stv_shown_composite is the candidate stv to compare against the satellite.
        # is_match=False: mismatch_negative_stv_composite is the (non-matching) candidate stv.
        #   We must NOT fall back to img["stv_composite"] for the False case — that is the query
        #   location's own real stv, which would trivially match the satellite and corrupt the task.
        #   The label is intentionally neutral ("Street-view image") for both cases so the model
        #   cannot infer the answer from the image description alone.
        if record.get("mismatch_is_match"):
            add("Street-view image", record.get("stv_shown_composite"))
        else:
            add("Street-view image", record.get("mismatch_negative_stv_composite"))
        return selected

    if topic.startswith("mismatch_mcq"):
        add("Marked satellite image", record.get("sat_marked_path"))
        add("Four candidate street-view grid", record.get("composite_stv_labeled_path"))
        return selected

    # normal questions
    add("Marked satellite image", img.get("satellite_marked"))
    add("Four labeled street-view angles", img.get("stv_composite_labeled"))

    return selected

def filter_images_for_ablation(images_used, ablation_mode):
    """
    Filters the images_used array based on the requested ablation mode.
    """
    if ablation_mode == "none" or not ablation_mode:
        return images_used

    filtered_images = []
    for img in images_used:
        path = img.get("path", "")
        
        is_satellite = "/sat/" in path or "/sat_marked/" in path or "/sat_arrow/" in path
        is_streetview = "/sv/" in path or "/composite/" in path

        if ablation_mode == "no_satellite" and is_satellite:
            continue 
            
        if ablation_mode == "no_streetview" and is_streetview:
            continue 

        filtered_images.append(img)
        
    return filtered_images

def skip_for_ablation(record, ablation_mode):
    if ablation_mode == "none" or not ablation_mode:
        return False

    topic = record.get("topic", "")

    if topic == "camera_direction":
        return True

    if topic.startswith("mismatch_"):
        return True

    # mismatch_mcq relies entirely on a street-view candidate grid;
    # skipping under no_streetview ablation keeps results consistent.
    if ablation_mode == "no_streetview" and topic.startswith("mismatch_mcq"):
        return True

    return False