"""modes.py — the single, unified image-ablation mode table for the suite.

Two earlier copies of this logic existed (reasoning_distill/run_modes_vllm.py and
run_ablation.py) and they disagreed on which modes apply to the 9 urban-attribute
topics. This is the one authoritative table.

An ABLATION MODE controls which of a record's images the model actually sees:

  full      everything the topic provides (sat + stv)        -> ALL topics
  blind     no images at all (text-only control)             -> ALL topics
  sat_only  satellite-perspective image(s) only              -> see applicability
  sv_only   street-view image(s) only                        -> see applicability

Applicability is topic-aware because dropping a view is only meaningful when the
two perspectives are SEPARATE droppable images:

  - mismatch_* (binary/mcq): sat and stv are separate -> full/sat_only/sv_only/blind
  - camera_direction: full/sat_only/blind. NO sv_only — the 4 arrow-satellite
    images ARE the answer options; dropping them is degenerate.
  - the 9 satellite_marked urban-attribute topics: DESIGNED dual-perspective
    (marked sat + 4 SV angles), so all four modes are meaningful — this is the
    "does the model fuse both views?" question.
  - any other single-perspective topic: full/blind only.
"""
from __future__ import annotations

from images import is_satellite_role, is_streetview_role

# Canonical execution/report order: most-degraded -> least (blind, then drop-one
# views, then full). `full` lands LAST so it reads as the payoff after the
# ablations. This ordering is the single source of truth — modes_for_topic()
# filters this list per topic, so applicability and order never drift. Order does
# NOT affect correctness (each mode is scored independently); only sequencing.
ALL_MODES = ["blind", "sv_only", "sat_only", "full"]

MISMATCH_TOPICS = {
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard",
}

# The 9 urban-attribute topics — designed dual-perspective (marked sat + 4 SV).
URBAN_ATTRIBUTE_TOPICS = {
    "land_use", "building_height", "urban_density", "junction_type",
    "green_space", "amenity_richness", "road_type", "road_surface",
    "transit_density",
}


def modes_for_topic(topic: str) -> list[str]:
    """Applicable modes for a topic, in canonical ALL_MODES order."""
    if topic in MISMATCH_TOPICS or topic in URBAN_ATTRIBUTE_TOPICS:
        applicable = {"blind", "sv_only", "sat_only", "full"}
    elif topic == "camera_direction":
        applicable = {"blind", "sat_only", "full"}  # no sv_only (option imgs are answer)
    else:
        applicable = {"blind", "full"}  # single-perspective fallback
    return [m for m in ALL_MODES if m in applicable]


def filter_images_for_mode(images: list[dict], mode: str) -> list[dict]:
    """Apply an ablation mode to the full {role,image} list from images.build_images."""
    if mode == "full":
        return list(images)
    if mode == "blind":
        return []
    if mode == "sat_only":
        return [im for im in images if not is_streetview_role(im["role"])]
    if mode == "sv_only":
        return [im for im in images if not is_satellite_role(im["role"])]
    raise ValueError(f"unknown mode {mode!r}")


def resolve_modes(topic: str, requested: list[str] | None) -> list[str]:
    """Intersection of a topic's applicable modes with the requested subset
    (None = all applicable). Preserves canonical mode order."""
    applicable = modes_for_topic(topic)
    if not requested:
        return applicable
    req = set(requested)
    return [m for m in applicable if m in req]
