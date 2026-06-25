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
        # Bounded set of indices known to fail (printed once per failure).
        # Prevents log spam when the same bad record is sampled repeatedly.
        self._bad_indices: set[int] = set()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # Known dataset gaps (São Paulo context, London Canary Wharf SV) +
        # malformed records (missing image_mode, wrong SV path count, None
        # paths) must not kill a dataloader worker — if a worker raises, the
        # whole training run dies. Fall forward to the next record instead.
        tries = 0
        cur = idx
        while tries < 10:
            try:
                return convert_record(self.records[cur], self.base_dir, self.max_edge)
            except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as e:
                if cur not in self._bad_indices:
                    self._bad_indices.add(cur)
                    print(f"[dataset] skipping record {cur} ({type(e).__name__}): {e}")
                cur = (cur + 1) % len(self.records)
                tries += 1
        raise RuntimeError(f"10 consecutive bad records starting at {idx} — dataset is broken")


# ---------------------------------------------------------------------------
# Token length measurement
# ---------------------------------------------------------------------------


def tokenize_converted(converted: dict, processor) -> int:
    """Tokenize a convert_record output through the processor, return seq length.

    Shared by measure_token_lengths and select_probe_samples — keeps both on the
    same code path so the selector ranks by the same number training will see.
    """
    images: list = []
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
    return int(inputs["input_ids"].shape[1])


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
        lengths.append(tokenize_converted(converted, processor))

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


# ---------------------------------------------------------------------------
# Worst-case sample selection for batch-size probing
# ---------------------------------------------------------------------------

# Upper bound of images injected into user_content by convert_record, keyed by
# image_mode. Matches the logic in convert_record exactly:
#   - satellite_arrow:      1 query_sv + 4 option sats        = 5
#   - streetview_composite: 1 sat_marked + 4 SV angles        = 5
#   - streetview_binary:    1 sat_marked + 4 negative SVs     = 5
#   - streetview_mega:      1 sat_marked + 1 mega composite   = 2
#   - satellite_marked/satellite_only (+fallbacks):           = 1
IMG_COUNT_BY_MODE = {
    "satellite_arrow":      5,
    "streetview_composite": 5,
    "streetview_binary":    5,
    "streetview_mega":      2,
    "satellite_marked":     1,
    "satellite_only":       1,
    "streetview_single":    1,
}


def _cheap_cost_score(record: dict) -> int:
    """Cheap O(1) score used as a coarse filter before per-sample tokenization.

    Returns a monotonic "expensive-looking" score combining image count
    (dominant factor) with text length (tiebreaker). No I/O, no decoding.
    """
    mode = record.get("image_mode", "satellite_only")
    n_img = IMG_COUNT_BY_MODE.get(mode, 1)
    q = record.get("question") or ""
    opts = record.get("options") or {}
    txt = len(q) + sum(len(v or "") for v in opts.values())
    # 10_000 multiplier makes image_count dominate for all realistic text sizes.
    return n_img * 10_000 + txt


def cost_sorted_indices(records: list[dict]) -> list[int]:
    """Return indices of `records` sorted by cheap cost score (descending).

    Used both by select_probe_samples (for the Stage-A filter) and by the
    fail-fast curriculum prefix in train.py — same score so the probe's
    worst case lines up with what training sees first.
    """
    return sorted(range(len(records)), key=lambda i: _cheap_cost_score(records[i]), reverse=True)


def build_synthetic_worst_case(max_edge: int, observed_max_text_len: int) -> dict:
    """Build an upper-bound probe sample — 5 max-edge blank images + padded text.

    Returns the same shape as convert_record(): {"messages": [...]}. This
    record is deliberately not drawn from the dataset — it is a provable
    ceiling. If the probe passes batch-size N on this, no real 5-image batch
    can OOM from image content at bs=N. Blank gray avoids triggering any
    content-dependent codepath in the processor.
    """
    # Padding the user text to the observed max length ensures token budget is
    # at least as large as anything real we've seen. Cap at 4096 chars so we
    # don't generate megabytes of placeholder text by accident.
    pad_len = min(max(observed_max_text_len, 512), 4096)
    # Use a neutral question that mimics the real format (4 MCQ options).
    base_q = "Which of the following best describes the scene?"
    options_text = "A. option A\nB. option B\nC. option C\nD. option D"
    padding = "x" * max(0, pad_len - len(base_q) - len(options_text) - 2)
    user_text = f"{base_q} {padding}\n{options_text}"

    # 5 blank RGB images at the post-resize edge — same size convert_record would
    # produce for 5-image heavy modes. Neutral gray (128) is safer than all-black
    # for any processor that special-cases saturated pixels.
    imgs = [Image.new("RGB", (max_edge, max_edge), (128, 128, 128)) for _ in range(5)]

    user_content: list[dict] = [{"type": "text", "text": user_text}]
    for img in imgs:
        user_content.append({"type": "image", "image": img})

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
        ]
    }


def select_probe_samples(
    records: list[dict],
    base_dir: str,
    max_edge: int,
    processor,
    k_real: int = 8,
    stage_a_topn: int = 200,
    include_synthetic: bool = True,
) -> tuple[list[dict], dict]:
    """Select worst-case probe samples for VRAM probing.

    Two-stage: (A) cheap image-count+text score across all records → top N;
    (B) tokenize top-N through the real processor → top K by true token count.
    Optionally append a synthetic upper-bound sample.

    Returns (samples, info) where `info` contains stats for logging:
        {
          "n_records": int,
          "stage_a_topn": int,
          "k_real": int,
          "token_lengths": list[int],       # for the selected K real samples
          "max_token_length": int,
          "synthetic_included": bool,
          "synthetic_token_length": int | None,
        }
    """
    n = len(records)
    if n == 0:
        raise ValueError("select_probe_samples called with no records")

    # Stage A — cheap filter
    stage_a_topn = min(stage_a_topn, n)
    ranked = cost_sorted_indices(records)[:stage_a_topn]

    # Stage B — tokenize, rank by real token count
    scored: list[tuple[int, int, dict]] = []  # (tok_len, original_idx, converted)
    observed_max_text = 0
    for idx in ranked:
        r = records[idx]
        observed_max_text = max(
            observed_max_text,
            len(r.get("question") or "") + sum(len(v or "") for v in (r.get("options") or {}).values()),
        )
        try:
            conv = convert_record(r, base_dir, max_edge)
            tok_len = tokenize_converted(conv, processor)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as e:
            # Bad records are fine to skip here — probe only needs K working ones.
            print(f"[probe/selector] skipping record {idx}: {type(e).__name__}: {e}")
            continue
        scored.append((tok_len, idx, conv))

    if not scored:
        raise RuntimeError(
            f"Probe sample selection found 0 tokenizable records out of "
            f"{stage_a_topn} candidates. Dataset is broken."
        )

    scored.sort(key=lambda t: -t[0])
    top_k = scored[:k_real]
    samples = [conv for (_, _, conv) in top_k]
    token_lengths = [tl for (tl, _, _) in top_k]

    info: dict = {
        "n_records": n,
        "stage_a_topn": stage_a_topn,
        "k_real": len(samples),
        "token_lengths": token_lengths,
        "max_token_length": token_lengths[0] if token_lengths else 0,
        "synthetic_included": False,
        "synthetic_token_length": None,
    }

    if include_synthetic:
        try:
            synth = build_synthetic_worst_case(max_edge, observed_max_text)
            synth_tok = tokenize_converted(synth, processor)
            samples.append(synth)
            info["synthetic_included"] = True
            info["synthetic_token_length"] = synth_tok
        except Exception as e:
            # Rare — some processor builds reject uniform images. Fall forward
            # with real-only probe; the top-K already covers the realistic worst.
            print(f"[probe/selector] synthetic sample skipped ({type(e).__name__}: {e})")

    return samples, info
