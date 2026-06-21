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

ALL_MODES = ["full", "sat_only", "sv_only", "blind"]

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
    if topic in MISMATCH_TOPICS:
        return ["full", "sat_only", "sv_only", "blind"]
    if topic == "camera_direction":
        return ["full", "sat_only", "blind"]  # no sv_only (option imgs are answer)
    if topic in URBAN_ATTRIBUTE_TOPICS:
        return ["full", "sat_only", "sv_only", "blind"]
    return ["full", "blind"]  # single-perspective fallback


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
