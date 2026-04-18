"""
data.py — Dataset loading, record conversion, and token measurement.

Image handling:
  - camera_direction (satellite_arrow): query street-view + 4 arrow-overlaid
    satellite images with corner A/B/C/D labels so the model can tell which
    image the letter refers to (burned-in, top-left, 10% of image width).
  - mismatch_binary (streetview_binary / streetview_composite): satellite_marked +
    4 individual SV images, each with a corner A/B/C/D label.
  - streetview_mega (mismatch_mcq): satellite_marked + mega composite grid (the
    grid itself already labels each cell).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from config import SEED, SYSTEM_PROMPT, find_dataset_dir, SPLIT

# Add composite_utils to path
_dataset_dir = find_dataset_dir()
sys.path.insert(0, str(_dataset_dir))
from composite_utils import (
    STV_ANGLES,
    get_images_for_question,
    make_sat_marked,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resize_image(img: Image.Image, max_edge: int) -> Image.Image:
    """Resize so longest edge <= max_edge, convert to RGB."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Corner A/B/C/D labels for multi-image options
# ---------------------------------------------------------------------------

# Cached font lookup — picked once per process.
_LABEL_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}

_LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _get_label_font(size: int) -> ImageFont.ImageFont:
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


def add_corner_label(img: Image.Image, letter: str) -> Image.Image:
    """Burn a small white-box A/B/C/D label into the top-left corner.

    Style: "V1 small white box" — 10% of image width, thin 2px black border,
    75%-of-box bold black letter. Chosen to be clearly visible while occupying
    minimal pixel area so model vision capacity isn't diverted to decoration.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    box_size = max(24, int(img.width * 0.10))
    pad = max(4, int(img.width * 0.015))
    x0, y0 = pad, pad
    x1, y1 = pad + box_size, pad + box_size
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    font = _get_label_font(max(12, int(box_size * 0.75)))
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x0 + (box_size - tw) // 2 - bbox[0]
    ty = y0 + (box_size - th) // 2 - bbox[1]
    draw.text((tx, ty), letter, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------


def convert_record(record: dict, base_dir: str, max_edge: int) -> dict:
    """Convert one JSONL record into the Unsloth/Qwen3.5 chat format.

    Returns {"messages": [...]} with typed content parts.
    Images are resized and converted to RGB.
    """
    result = get_images_for_question(record, base_dir=base_dir)
    mode = record.get("image_mode", "satellite_only")

    # Format question + options
    opts = record["options"]
    text = record["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())

    user_content: list[dict] = [{"type": "text", "text": text}]

    if mode == "satellite_arrow":
        # camera_direction: query street-view + 4 arrow-overlaid satellites.
        # The query SV is the photographer's viewpoint; the 4 sats each show the
        # same location with a red arrow in a different direction. Without the
        # query SV the task is unanswerable. Each option sat gets a burned-in
        # A/B/C/D label so the model can bind the letter to the image directly
        # rather than relying on positional ordering in the content stream.
        query_sv = result.get("query_sv")
        if query_sv is not None:
            user_content.append({
                "type": "image",
                "image": resize_image(query_sv, max_edge),
            })
        if result.get("options"):
            for letter in ("A", "B", "C", "D"):
                if letter in result["options"]:
                    sized = resize_image(result["options"][letter], max_edge)
                    user_content.append({
                        "type": "image",
                        "image": add_corner_label(sized, letter),
                    })

    elif mode in ("streetview_composite", "streetview_binary"):
        # mismatch_binary: satellite_marked + 4 individual SV images. The SV
        # images are in cardinal-angle order (along_fwd / along_bwd / cross_left /
        # cross_right) — we burn A/B/C/D corner labels so the model can refer to
        # specific SV angles unambiguously.
        sat_path = os.path.join(base_dir, record["images"]["satellite"])
        sat_marked = make_sat_marked(sat_path)
        user_content.append({"type": "image", "image": resize_image(sat_marked, max_edge)})

        if mode == "streetview_composite":
            sv_iter = [
                (angle, os.path.join(base_dir, record["images"][f"streetview_{angle}"]))
                for angle in STV_ANGLES
            ]
        else:
            # streetview_binary (match=False): negative location's 4 SV images
            neg_paths = record.get("mismatch_negative_stv_paths") or []
            sv_iter = [(None, os.path.join(base_dir, p)) for p in neg_paths]

        for (_angle, sv_path), letter in zip(sv_iter, ("A", "B", "C", "D")):
            sv_img = Image.open(sv_path)
            sized = resize_image(sv_img, max_edge)
            user_content.append({"type": "image", "image": add_corner_label(sized, letter)})

    elif mode == "streetview_mega":
        # mismatch_mcq: satellite_marked + mega composite (keeps the grid — MCQ needs it)
        sat_path = os.path.join(base_dir, record["images"]["satellite"])
        sat_marked = make_sat_marked(sat_path)
        user_content.append({"type": "image", "image": resize_image(sat_marked, max_edge)})
        user_content.append({"type": "image", "image": resize_image(result["primary"], max_edge)})

    else:
        # satellite_marked, satellite_only: single image
        user_content.append({"type": "image", "image": resize_image(result["primary"], max_edge)})

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]},
        ]
    }


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class EollmDataset:
    """On-demand converting dataset for EOLLM vision SFT."""

    def __init__(self, records: list[dict], base_dir: str, max_edge: int):
        self.records = records
        self.base_dir = base_dir
        self.max_edge = max_edge

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return convert_record(self.records[idx], self.base_dir, self.max_edge)


# ---------------------------------------------------------------------------
# Token length measurement
# ---------------------------------------------------------------------------


def measure_token_lengths(
    records: list[dict],
    base_dir: str,
    max_edge: int,
    processor,
    n_samples: int = 300,
) -> dict:
    """Measure token length distribution using the actual Qwen3.5 processor.

    Returns dict with p50, p90, p95, p99, max, recommended max_seq_length.
    """
    import random
    rng = random.Random(SEED)
    sample_indices = rng.sample(range(len(records)), min(n_samples, len(records)))

    lengths = []
    for i, idx in enumerate(sample_indices):
        converted = convert_record(records[idx], base_dir, max_edge)

        images = []
        messages = []
        for msg in converted["messages"]:
            content_parts = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    content_parts.append({"type": "image"})
                    images.append(part["image"])
            messages.append({"role": msg["role"], "content": content_parts})

        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        img_input = images[0] if len(images) == 1 else images
        inputs = processor(img_input, text, add_special_tokens=False, return_tensors="pt")
        lengths.append(inputs["input_ids"].shape[1])

        if (i + 1) % 50 == 0:
            print(f"  measured {i+1}/{len(sample_indices)} samples...")

    lengths.sort()
    n = len(lengths)
    result = {
        "n_samples": n,
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.90)],
        "p95": lengths[int(n * 0.95)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }

    # p99 rounded up, with safe floor of 8192
    p99 = result["p99"]
    boundaries = [2048, 4096, 8192, 16384, 32768]
    recommended = next((b for b in boundaries if b >= p99), 32768)
    recommended = max(recommended, 8192)
    result["recommended_max_seq_length"] = recommended
    return result
