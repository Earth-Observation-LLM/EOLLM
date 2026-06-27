"""images.py — canonical, training-identical image construction for the suite.

The benchmark feeds a vision-language model a specific set of images per record,
and the EXACT pixels matter: a marked satellite, four corner-labelled street-view
angles, arrow overlays, mega-composite grids. Getting this wrong silently
corrupts every downstream number, so this module is the SINGLE source of truth
for "which images does record R get, and what do they look like".

It is a self-contained re-implementation of `training/data.py:convert_record`'s
image branches, dispatched on the record's `image_mode`. The actual pixel
operations (marked-satellite rendering, mega tiling, the angle/letter corner
labels) are delegated to the dataset's OWN `composite_utils` so they are
byte-identical to what training and the existing evals produced — this is data,
not another pipeline, so importing it by path keeps the suite decoupled from all
eval code while guaranteeing pixel parity. `tests/test_image_parity.py` asserts
that parity against training/data.py and fails loudly if it ever drifts.

Returned structure for one record:

    [
      {"role": "satellite_marked", "image": <PIL.Image>},
      {"role": "streetview_along_fwd", "image": <PIL.Image>},   # labelled "Fwd"
      ...
    ]

`role` is a stable, human-meaningful label used by the prompt, the ablation
filter (sat vs sv), the attention image-span labelling, and the viewer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- locate the dataset and its composite_utils (the authoritative pixel ops) --
# We import composite_utils by path. It is part of the DATASET (it builds the
# dataset's own imagery), not part of any eval pipeline, so this respects the
# "zero eval-code imports" boundary while guaranteeing the pixels match training.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_dataset_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("DATASET_DIR")
    if env:
        return Path(env)
    cand = _REPO_ROOT / "dataset_content" / "EODATA_compressed_final"
    if (cand / "composite_utils.py").exists():
        return cand
    raise FileNotFoundError(
        f"Cannot find composite_utils.py under {cand}. Set DATASET_DIR."
    )


_DATASET_DIR = _resolve_dataset_dir()
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from composite_utils import (  # noqa: E402  (path-injected import)
    STV_ANGLES,
    get_images_for_question,
    make_sat_marked,
)

# Canonical viewing-direction labels burned into the four street-view angles for
# satellite_marked. They are NOT A/B/C/D, because in satellite_marked the
# A/B/C/D letters are the ANSWER CHOICES — a letter on the image would collide
# with the answer. (Mirrors training/data.py exactly.)
_ANGLE_LABEL = {
    "along_fwd": "Fwd",
    "along_bwd": "Bwd",
    "cross_left": "Left",
    "cross_right": "Right",
}

# Street-view angle gate (satellite_marked only) — MIRRORS training/data.py.
# SV_ANGLES restricts which of the 4 street-view angles are fed for
# satellite_marked urban-attribute records. Default = all 4 (no behavior
# change, so every existing eval is byte-identical). Set SV_ANGLES=along_fwd to
# feed marked-satellite + forward SV only — required to eval a model TRAINED
# with SV_ANGLES=along_fwd on its own image distribution. The gate runs BEFORE
# the ablation filter in modes.py, so `full`/`sv_only` compose correctly on top
# (sv_only then means "forward SV only" for such a model). Comma-separated for
# any subset, e.g. SV_ANGLES=along_fwd,cross_left.
_SV_ANGLES_ENV = os.environ.get("SV_ANGLES", "").strip()
SV_ANGLES_KEEP = (
    [a.strip() for a in _SV_ANGLES_ENV.split(",") if a.strip()]
    if _SV_ANGLES_ENV
    else list(STV_ANGLES)
)
# Fail loud on a typo (e.g. SV_ANGLES=fwd) rather than silently drop every SV
# image and eval satellite-only without anyone noticing.
_BAD_ANGLES = [a for a in SV_ANGLES_KEEP if a not in STV_ANGLES]
if _BAD_ANGLES:
    raise ValueError(
        f"SV_ANGLES contains unknown angle(s) {_BAD_ANGLES}. "
        f"Valid angles: {list(STV_ANGLES)}"
    )


# ---------------------------------------------------------------------------
# Resize + corner-label — byte-identical re-implementation of training/data.py
# ---------------------------------------------------------------------------

def resize_image(img: Image.Image, max_edge: int) -> Image.Image:
    """Resize so the longest edge <= max_edge, convert to RGB. (== data.py)"""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


_LABEL_FONT_CACHE: dict[int, "ImageFont.ImageFont"] = {}
_LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _get_label_font(size: int):
    if size not in _LABEL_FONT_CACHE:
        font = None
        for path in _LABEL_FONT_CANDIDATES:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        if font is None:
            font = ImageFont.load_default()
        _LABEL_FONT_CACHE[size] = font
    return _LABEL_FONT_CACHE[size]


def add_corner_label(img: Image.Image, label: str) -> Image.Image:
    """Burn a small white-box label into the top-left corner. (== data.py V1.)

    box height = 10% of image width, 2px black border, bold black text at 75%
    of box height. Single-char labels stay square; word labels (Fwd/Bwd/...)
    expand the box width.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    box_h = max(24, int(img.width * 0.10))
    pad = max(4, int(img.width * 0.015))
    font = _get_label_font(max(12, int(box_h * 0.75)))

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    inner_pad = max(4, int(box_h * 0.20))
    box_w = max(box_h, tw + 2 * inner_pad)

    x0, y0 = pad, pad
    x1, y1 = pad + box_w, pad + box_h
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    tx = x0 + (box_w - tw) // 2 - bbox[0]
    ty = y0 + (box_h - th) // 2 - bbox[1]
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Per-record image construction — dispatch on image_mode (== data.py branches)
# ---------------------------------------------------------------------------

def build_images(record: dict, base_dir: str | Path, max_edge: int) -> list[dict]:
    """Return the full list of {role, image} this record feeds the model.

    Pixel-identical to training/data.py:convert_record for every image_mode.
    `base_dir` is the data root the record's relative paths resolve against
    (e.g. the benchmark/ dir). No ablation filtering here — that is modes.py's
    job, applied to this full list.
    """
    base = str(base_dir)
    mode = record.get("image_mode", "satellite_only")
    result = get_images_for_question(record, base_dir=base)
    out: list[dict] = []

    def push(role: str, img: Image.Image, label: str | None = None):
        sized = resize_image(img, max_edge)
        if label is not None:
            sized = add_corner_label(sized, label)
        out.append({"role": role, "image": sized})

    if mode == "satellite_arrow":
        # camera_direction: query street-view + 4 arrow-overlaid satellites
        # (each gets a burned-in A/B/C/D corner label — here the letters ARE the
        # option selectors, so labelling the option images is correct).
        query_sv = result.get("query_sv")
        if query_sv is not None:
            push("query_streetview", query_sv)
        opts = result.get("options") or {}
        for letter in ("A", "B", "C", "D"):
            if letter in opts and opts[letter] is not None:
                push(f"option_{letter}_satellite_arrow", opts[letter], label=letter)

    elif mode in ("streetview_composite", "streetview_binary"):
        # mismatch_binary: marked satellite + 4 individual SV images, each with
        # an A/B/C/D corner label (here the labels select the SV the model
        # reasons about).
        sat_path = os.path.join(base, record["images"]["satellite"])
        push("satellite_marked", make_sat_marked(sat_path))
        if mode == "streetview_composite":
            sv_iter = [
                os.path.join(base, record["images"][f"streetview_{a}"])
                for a in STV_ANGLES
            ]
        else:
            sv_iter = list(record.get("mismatch_negative_stv_paths") or [])
            sv_iter = [p if os.path.isabs(p) else os.path.join(base, p) for p in sv_iter]
        for sv_path, letter in zip(sv_iter, ("A", "B", "C", "D")):
            push(f"streetview_option_{letter}", Image.open(sv_path), label=letter)

    elif mode == "streetview_mega":
        # mismatch_mcq: marked satellite + mega composite grid (grid self-labels).
        sat_path = os.path.join(base, record["images"]["satellite"])
        push("satellite_marked", make_sat_marked(sat_path))
        push("streetview_mega", result["primary"])

    elif mode == "satellite_marked":
        # Urban-attribute questions: marked satellite + 4 SV angles labelled by
        # DIRECTION (Fwd/Bwd/Left/Right) — NOT A/B/C/D (those are the answers).
        sat_path = os.path.join(base, record["images"]["satellite"])
        push("satellite_marked", make_sat_marked(sat_path))
        # SV_ANGLES_KEEP gates which angles are fed (default all 4). Iterate in
        # canonical STV_ANGLES order so kept images keep a stable order
        # regardless of how SV_ANGLES was written.
        for angle in (a for a in STV_ANGLES if a in SV_ANGLES_KEEP):
            rel = record["images"].get(f"streetview_{angle}")
            if not rel:
                continue  # rare missing angle (London SV gaps) — skip, don't fail
            push(f"streetview_{angle}", Image.open(os.path.join(base, rel)),
                 label=_ANGLE_LABEL[angle])

    else:
        # satellite_only: a single raw satellite, no SV available.
        push("satellite", result["primary"])

    return out


# ---------------------------------------------------------------------------
# Role classification for ablation (sat vs sv) — used by modes.py
# ---------------------------------------------------------------------------

def is_satellite_role(role: str) -> bool:
    return "satellite" in role or "sat_marked" in role or "sat_arrow" in role


def is_streetview_role(role: str) -> bool:
    return "streetview" in role or "stv" in role
