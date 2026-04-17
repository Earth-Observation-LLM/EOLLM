"""
data.py — Dataset loading, record conversion, and token measurement.

Image handling corrections (from domain expert):
  - camera_direction: show ONLY the 4 satellite-with-arrow option images (no street view)
  - mismatch_binary: show satellite_marked + 4 individual SV images (not composited into grid)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from PIL import Image

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
        # camera_direction: show ONLY the 4 satellite-with-arrow option images
        # Domain expert correction: no street view query image
        if result.get("options"):
            for letter in ("A", "B", "C", "D"):
                if letter in result["options"]:
                    user_content.append({
                        "type": "image",
                        "image": resize_image(result["options"][letter], max_edge),
                    })

    elif mode in ("streetview_composite", "streetview_binary"):
        # mismatch_binary: satellite_marked + 4 individual SV images
        # Domain expert correction: individual images, NOT composited into a grid
        sat_path = os.path.join(base_dir, record["images"]["satellite"])
        sat_marked = make_sat_marked(sat_path)
        user_content.append({"type": "image", "image": resize_image(sat_marked, max_edge)})

        if mode == "streetview_composite":
            # match=True or match=False with own SV: show query location's 4 angles
            for angle in STV_ANGLES:
                sv_path = os.path.join(base_dir, record["images"][f"streetview_{angle}"])
                sv_img = Image.open(sv_path)
                user_content.append({"type": "image", "image": resize_image(sv_img, max_edge)})
        else:
            # streetview_binary (match=False): show negative location's 4 SV images
            neg_paths = record.get("mismatch_negative_stv_paths") or []
            for path in neg_paths:
                sv_img = Image.open(os.path.join(base_dir, path))
                user_content.append({"type": "image", "image": resize_image(sv_img, max_edge)})

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
