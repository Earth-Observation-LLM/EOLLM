"""Benchmark dataset loader + image resolution + missing-image accounting."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image


BENCHMARK_ROOT = Path("/home/ain480/evaluation/EarthMLLMEval/data")
BENCHMARK_FILE = BENCHMARK_ROOT / "benchmark_with_answers.jsonl"


def load_benchmark(path: Path = BENCHMARK_FILE) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)
    return records


def resolve_image_paths(record: dict, root: Path = BENCHMARK_ROOT) -> dict[str, Optional[Path]]:
    """Resolve every image referenced by a record to an absolute path.

    Returns a dict keyed by the image's role (e.g. "satellite",
    "streetview_along_fwd"). Values are absolute Path objects, or None if the
    record didn't have that image. Files that DO NOT EXIST on disk are
    returned as None and counted in `missing_images` of the caller.
    """
    images = record.get("images") or {}
    out: dict[str, Optional[Path]] = {}
    for role, val in images.items():
        if not isinstance(val, str):
            # e.g. streetview_road_bearing (float), date strings — skip non-paths.
            continue
        if not (val.endswith(".png") or val.endswith(".jpg") or val.endswith(".jpeg")):
            continue
        p = root / val
        out[role] = p if p.exists() else None
    return out


def select_images_for_record(
    record: dict,
    mode: str = "sat+stv4",
    root: Path = BENCHMARK_ROOT,
) -> tuple[list[Image.Image], list[str], int]:
    """Choose which images to feed the model.

    mode:
      - "sat+stv4": satellite + 4 street-view (default; richest input).
                   For mismatch_* topics, uses task-specific images instead.
      - "sat_only": satellite only.
      - "stv_only": street-view only.

    Returns (PIL_images, role_labels, n_missing).
    """
    resolved = resolve_image_paths(record, root)
    topic = record.get("topic", "")

    paths: list[tuple[str, Optional[Path]]] = []

    # Mismatch topics use task-specific image fields (option_sat_paths etc.)
    # The benchmark already lists the relevant images under "images"; we just
    # need to make sure we pass the right ones. For mismatch, the question is
    # whether the streetview_shown matches one of N satellite options. Use
    # whatever the dataset provides under "stv_shown_*" + "option_sat_paths".
    if topic.startswith("mismatch"):
        # Pull from record-level fields (not just images dict).
        if record.get("stv_shown_composite"):
            p = root / record["stv_shown_composite"]
            paths.append(("stv_shown_composite", p if p.exists() else None))
        elif record.get("composite_stv_path"):
            p = root / record["composite_stv_path"]
            paths.append(("composite_stv_path", p if p.exists() else None))
        # Option satellites (multi-choice mismatch_mcq) or single sat (binary).
        opt_paths = record.get("option_sat_paths") or []
        for i, op in enumerate(opt_paths):
            if isinstance(op, str):
                p = root / op
                paths.append((f"option_sat_{i}", p if p.exists() else None))
        if not opt_paths:
            # Binary mismatch: just need the satellite + streetview.
            if "satellite" in resolved:
                paths.append(("satellite", resolved["satellite"]))
    else:
        if mode in ("sat+stv4", "sat_only") and "satellite" in resolved:
            paths.append(("satellite", resolved["satellite"]))
        if mode in ("sat+stv4", "stv_only"):
            for role in [
                "streetview_along_fwd",
                "streetview_along_bwd",
                "streetview_cross_left",
                "streetview_cross_right",
            ]:
                if role in resolved:
                    paths.append((role, resolved[role]))

    # For camera_direction, the question often references the satellite-marked
    # image showing a query bearing. Use that when available.
    if topic == "camera_direction":
        if record.get("sat_marked_path"):
            p = root / record["sat_marked_path"]
            if p.exists():
                paths.insert(0, ("sat_marked", p))
        if record.get("query_stv_path"):
            p = root / record["query_stv_path"]
            if p.exists():
                paths.append(("query_stv", p))

    images: list[Image.Image] = []
    roles: list[str] = []
    n_missing = 0
    for role, p in paths:
        if p is None:
            n_missing += 1
            continue
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            roles.append(role)
        except Exception:
            n_missing += 1
    return images, roles, n_missing


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in analyzing satellite and street-view imagery of urban "
    "and natural environments. Answer the multiple-choice question by replying "
    "with ONLY the letter of the best option (A, B, C, or D). Do not explain."
)


def build_prompt(record: dict) -> str:
    """Build the textual prompt (without images) for a record.

    Identical for every model so accuracy is comparable. The vision-token
    placement is handled by each model's processor.
    """
    q = record["question"]
    opts = record["options"]
    # Stable order: alphabetic.
    opt_lines = "\n".join(f"{L}. {opts[L]}" for L in sorted(opts.keys()))
    return f"{q}\n\n{opt_lines}\n\nAnswer with ONLY the letter of the correct option."


def valid_letters(record: dict) -> set[str]:
    return set(record["options"].keys())
