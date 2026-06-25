"""prompts.py — per-task steering prompt bank for the solve ladder.

The solve ladder escalates: greedy with the NEUTRAL prompt first (identical to the
benchmark), then, if that fails, the same question with a per-topic STEERING
instruction that points the model at the feature the question is about — and then
temperature.

What steering is and isn't:
  - It directs *where to look* ("focus on the road junction geometry"), which is
    legitimate: same images, same options, no answer revealed. The point of this
    branch is precisely that the model CAN answer from the right view; a steering
    prompt that nudges it toward that view is fair, and we report its use openly.
  - It must NEVER hint the answer, name an option, or describe the correct class.
    `assert_no_leak` below guards that: a steering string may not contain any
    option VALUE from any record of its topic (checked at load against the data).

A variant is a (name, system_suffix, user_prefix) triple. The base benchmark
system/user text comes from benchmark_suite.prompt; we only APPEND a steering
sentence so the answer format constraint (single letter) is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str            # "neutral" | "steer"
    steer: str           # appended to the system prompt; "" for neutral


# Neutral = exactly the benchmark. Always tried first (greedy).
NEUTRAL = Variant(name="neutral", steer="")

# One steering sentence per urban topic. Points at the discriminative visual
# feature WITHOUT naming any answer class. Keep these generic — they describe what
# to attend to, never what to conclude.
_STEER = {
    "junction_type":
        "Look carefully at how the roads meet in the street-level views: the number "
        "of approaching roads, their layout where they join, and any traffic-control "
        "or level-change cues. Decide which single street-view angle best shows the "
        "place where the roads meet before answering.",
    "road_type":
        "Look at the carriageway in the street-level views: its width, number of "
        "lanes, lane markings, central separation, and roadside context. Base your "
        "answer on the angle that most clearly shows the road itself.",
    "road_surface":
        "Look closely at the texture and material of the road surface itself in the "
        "street-level views — how smooth, patterned, or rough it appears underfoot — "
        "using the angle that shows the carriageway surface most clearly.",
    "building_height":
        "Look at the buildings in the street-level views and judge how many storeys "
        "they have, using vertical cues (windows, floors, relative scale) in the "
        "angle that best shows building facades.",
    "land_use":
        "Look at what the area is used for in the street-level views — the mix of "
        "buildings, open space, greenery, and activity — using whichever angles "
        "reveal the dominant land use.",
    "urban_density":
        "Look at how densely built-up the surroundings are in the street-level "
        "views — building spacing, footprint coverage, and open vs. built area — "
        "across the angles that show the broader setting.",
    "amenity_richness":
        "Look for amenities and points of interest in the street-level views — "
        "shops, services, signage, street furniture — using the angles that show "
        "the most activity and frontage.",
    "transit_density":
        "Look for public-transport infrastructure in the street-level views — stops, "
        "shelters, rails, dedicated lanes, overhead lines — using whichever angle "
        "shows transit features most clearly.",
}


def variants_for(topic: str) -> list[Variant]:
    """Ladder variants for a topic: neutral first, then steered (if defined)."""
    out = [NEUTRAL]
    s = _STEER.get(topic)
    if s:
        out.append(Variant(name="steer", steer=s))
    return out


def assert_no_leak(records: list[dict]) -> None:
    """Fail loudly if any steering string contains an option VALUE for its topic.

    Catches the one way steering could become cheating: accidentally naming an
    answer class. Run once at collector startup against the loaded dataset.
    """
    from collections import defaultdict
    values_by_topic: dict[str, set[str]] = defaultdict(set)
    for r in records:
        t = r.get("topic")
        for v in (r.get("options") or {}).values():
            if isinstance(v, str):
                values_by_topic[t].add(v.strip().lower())
    for topic, steer in _STEER.items():
        low = steer.lower()
        for val in values_by_topic.get(topic, ()):
            # word-ish containment: a full option value appearing in the steer text
            if val and len(val) > 3 and val in low:
                raise AssertionError(
                    f"steering prompt for {topic!r} leaks option value {val!r}")
