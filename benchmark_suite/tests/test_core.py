"""test_core.py — sanity-check the non-model core: modes, prompt, parsing, metrics.

No GPU / model needed. Validates the wiring that every backend depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE))

import modes
import prompt
import parsing
import metrics


def test_modes():
    assert modes.modes_for_topic("mismatch_mcq_hard") == ["full", "sat_only", "sv_only", "blind"]
    assert modes.modes_for_topic("camera_direction") == ["full", "sat_only", "blind"]
    assert "sv_only" not in modes.modes_for_topic("camera_direction")
    assert modes.modes_for_topic("land_use") == ["full", "sat_only", "sv_only", "blind"]
    assert modes.modes_for_topic("some_single_perspective_topic") == ["full", "blind"]
    # resolve_modes intersects with a request, preserving canonical order
    assert modes.resolve_modes("land_use", ["blind", "full"]) == ["full", "blind"]

    imgs = [
        {"role": "satellite_marked", "image": None},
        {"role": "streetview_along_fwd", "image": None},
        {"role": "streetview_cross_left", "image": None},
    ]
    assert len(modes.filter_images_for_mode(imgs, "full")) == 3
    assert len(modes.filter_images_for_mode(imgs, "blind")) == 0
    assert [i["role"] for i in modes.filter_images_for_mode(imgs, "sat_only")] == ["satellite_marked"]
    assert len(modes.filter_images_for_mode(imgs, "sv_only")) == 2
    print("modes OK")


def test_prompt():
    rec = {
        "question": "What is the land use?",
        "options": {"A": "Residential", "B": "Industrial", "C": "Park", "D": "Water"},
    }
    t = prompt.build_user_text(rec, ["satellite_marked", "streetview_along_fwd"])
    assert "Image 1: satellite_marked" in t
    assert "Image 2: streetview_along_fwd" in t
    assert "A. Residential" in t and "D. Water" in t
    assert "ONLY one letter (A or B or C or D)" in t
    # blind: no image lines
    t2 = prompt.build_user_text(rec, [])
    assert "No images are provided." in t2
    print("prompt OK")


def test_parsing():
    assert parsing.parse_letter("B") == "B"
    assert parsing.parse_letter("The answer is C.") == "C"
    assert parsing.parse_letter("Based on the view") is None  # no isolated letter
    assert parsing.parse_letter("(A)") == "A"
    p = parsing.parse_answer("I cannot tell", options={"A": "x", "B": "y"})
    assert p.refused and p.letter is None
    p2 = parsing.parse_answer("Industrial", options={"A": "Residential", "B": "Industrial"})
    assert p2.letter == "B"  # echoed option value -> letter
    print("parsing OK")


def test_metrics():
    rows = [
        {"gold": "A", "prediction": "A", "topic": "land_use", "difficulty": "easy",
         "options": {"A": "x", "B": "y", "C": "z", "D": "w"}},
        {"gold": "B", "prediction": "A", "topic": "land_use", "difficulty": "hard",
         "options": {"A": "x", "B": "y", "C": "z", "D": "w"}},
        {"gold": "C", "prediction": "C", "topic": "road_type", "difficulty": "easy",
         "options": {"A": "x", "B": "y", "C": "z", "D": "w"}},
    ]
    rep = metrics.build_full_report(rows)
    assert rep["overall"]["total"] == 3
    assert rep["overall"]["correct"] == 2
    assert abs(rep["overall"]["accuracy"] - 2 / 3) < 1e-9
    assert "land_use" in rep["by_topic"] and "road_type" in rep["by_topic"]
    print("metrics OK (sklearn:", metrics._HAVE_SKLEARN, ")")


if __name__ == "__main__":
    test_modes()
    test_prompt()
    test_parsing()
    test_metrics()
    print("\nALL CORE TESTS PASSED")
