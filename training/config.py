"""
config.py — GPU profiles, env overrides, paths, and run directory management.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Base model: prefer repo-local copy (portable across machines, no HF download needed).
# If the local copy is absent we FAIL — silent HF hub fallback risks snapshot drift
# between local and remote runs. Set BASE_MODEL explicitly to override.
#
# For multi-model runs (27B, 31B, Gemma) BASE_MODEL is set explicitly to an HF
# hub id (e.g. unsloth/Qwen3.5-27B) by launch_multimodel.sh — the model lives in
# the HF cache, not the repo-local models/ dir.
_LOCAL_MODEL = PROJECT_ROOT / "models" / "Qwen3.5-4B"
_BASE_MODEL_ENV = os.environ.get("BASE_MODEL")
if _BASE_MODEL_ENV:
    BASE_MODEL = _BASE_MODEL_ENV
elif _LOCAL_MODEL.exists():
    BASE_MODEL = str(_LOCAL_MODEL)
else:
    # Never silently fall back to the hub — forces user to make an explicit choice.
    # If hub download is truly wanted, set BASE_MODEL=unsloth/Qwen3.5-4B explicitly.
    raise FileNotFoundError(
        f"Base model not found at {_LOCAL_MODEL}. "
        f"Set BASE_MODEL env var to an explicit path (local dir or HF hub id). "
        f"Silent HF fallback is disabled to prevent snapshot drift between runs."
    )

# ---------------------------------------------------------------------------
# Model family — drives chat-template choice + collator label-masking strings
# ---------------------------------------------------------------------------
#
# Qwen3.5 and Gemma4 use DIFFERENT turn markers. train_on_responses_only masks
# everything that is not inside the response part, so these strings MUST match
# the model's actual chat template or label masking silently breaks (the run
# then trains on the whole prompt, or on nothing — check_label_masking() in
# train.py aborts loudly if that happens, which is the safety net).
#
# Verified on lab-ws against the real processors (2026-06):
#   Qwen3.5-27B : <|im_start|>user\n      / <|im_start|>assistant\n
#   gemma-4-*   : <|turn>user\n           / <|turn>model\n
# NB: this Gemma-4 build uses `<|turn>` markers, NOT the classic
# `<start_of_turn>` from Gemma 2/3.
#
# MODEL_FAMILY is set explicitly by the launcher. If unset we infer from the
# BASE_MODEL string so single-model runs (plain `python train.py` on the 4B)
# keep working with zero extra env.
# chat_template = None means "use the model's OWN shipped chat_template.jinja".
# We deliberately do NOT call unsloth get_chat_template() for Gemma: its bundled
# "gemma-4" template has a `list + str` bug on the system turn (crashes when
# system content is a list of parts), whereas the model's shipped template
# handles BOTH string and list content AND renders the identical <|turn> markers
# our collator masking strings target. Verified on lab-ws (2026-06). So None for
# both families now — each uses its own correct shipped template.
_FAMILY_MASKING = {
    "qwen":  {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n", "chat_template": None},
    "gemma": {"instruction_part": "<|turn>user\n",      "response_part": "<|turn>model\n",          "chat_template": None},
}


def _infer_family(base_model: str) -> str:
    low = base_model.lower()
    if "gemma" in low:
        return "gemma"
    if "qwen" in low:
        return "qwen"
    # Unknown → assume qwen (the validated default). Surface a warning rather
    # than guess silently; check_label_masking() will catch a real mismatch.
    print(f"[config] WARNING: could not infer model family from BASE_MODEL={base_model!r}; "
          f"defaulting to 'qwen'. Set MODEL_FAMILY=qwen|gemma to be explicit.")
    return "qwen"


MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "").strip().lower() or _infer_family(BASE_MODEL)
if MODEL_FAMILY not in _FAMILY_MASKING:
    raise ValueError(f"Unknown MODEL_FAMILY={MODEL_FAMILY!r}. Known: {sorted(_FAMILY_MASKING)}")

# Display name for W&B config + summary.txt. Defaults to the basename of the
# base model path so logs read "gemma-4-31B-it" instead of a generic label.
MODEL_NAME = os.environ.get("MODEL_NAME") or Path(BASE_MODEL).name

COLLATOR_INSTRUCTION_PART = _FAMILY_MASKING[MODEL_FAMILY]["instruction_part"]
COLLATOR_RESPONSE_PART    = _FAMILY_MASKING[MODEL_FAMILY]["response_part"]
CHAT_TEMPLATE             = _FAMILY_MASKING[MODEL_FAMILY]["chat_template"]

# SAVE_MERGED=1 writes a full merged 16-bit model (~52-62 GB for 27B/31B).
# Default OFF — the eval path (eval_base.py with EVAL_ADAPTER) loads base+adapter
# directly, so the merged model is never needed for evaluation and only burns
# disk. The LoRA adapter (~200 MB) is always saved regardless.
SAVE_MERGED = os.environ.get("SAVE_MERGED", "0") == "1"

# ---------------------------------------------------------------------------
# GPU profiles — dataset-independent knobs only
# ---------------------------------------------------------------------------

PROFILES = {
    "rtx_5090_32gb":       dict(vram_gb=32,  image_max_edge=768,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=4,  finetune_vision_layers=True),
    "rtx_4090_24gb":       dict(vram_gb=24,  image_max_edge=640,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=2,  finetune_vision_layers=True),
    "rtx_3090_24gb":       dict(vram_gb=24,  image_max_edge=512,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=2,  finetune_vision_layers=True),
    "a100_80gb":           dict(vram_gb=80,  image_max_edge=1024, lora_r=32, lora_alpha=32, lr=1e-4, initial_batch_guess=8,  finetune_vision_layers=True),
    "h100_80gb":           dict(vram_gb=80,  image_max_edge=1024, lora_r=32, lora_alpha=32, lr=1e-4, initial_batch_guess=8,  finetune_vision_layers=True),
    "h200_141gb":          dict(vram_gb=141, image_max_edge=1024, lora_r=64, lora_alpha=64, lr=1e-4, initial_batch_guess=16, finetune_vision_layers=True),
    # 96 GB physical; probe uses BATCH_SAFETY_FRAC (default 0.80) against FREE
    # VRAM to pick bs. initial_batch_guess=16 caps the probe at 32 (guess*2),
    # so the upper probe tier is actually explored on this card instead of
    # being capped at 24. lora_r=16 matches the 5090 recipe.
    "rtx_pro_6000_96gb":   dict(vram_gb=80,  image_max_edge=1024, lora_r=16, lora_alpha=16, lr=2.5e-4, initial_batch_guess=16, finetune_vision_layers=True),
    "safe_16gb":           dict(vram_gb=16,  image_max_edge=448,  lora_r=8,  lora_alpha=8,  lr=2e-4, initial_batch_guess=1,  finetune_vision_layers=False),
}


def _apply_profile_overrides(name: str, profile: dict) -> dict:
    """Return a COPY of `profile` with per-knob env overrides applied.

    The multi-model sweep pins image_max_edge=512 and lora_r/alpha=16 across all
    models regardless of which GPU profile is detected, without editing the
    PROFILES table. Returns a fresh dict so the shared PROFILES entry is never
    mutated (detect_profile used to return the shared dict by reference).

    Overrides (all optional):
      IMAGE_MAX_EDGE  -> image_max_edge
      LORA_R          -> lora_r
      LORA_ALPHA      -> lora_alpha   (defaults to LORA_R if only LORA_R set)
    """
    p = dict(profile)
    edge = os.environ.get("IMAGE_MAX_EDGE")
    if edge:
        p["image_max_edge"] = int(edge)
    r = os.environ.get("LORA_R")
    if r:
        p["lora_r"] = int(r)
        # If alpha not separately given, mirror r (alpha==r is the recipe).
        p["lora_alpha"] = int(os.environ.get("LORA_ALPHA", r))
    elif os.environ.get("LORA_ALPHA"):
        p["lora_alpha"] = int(os.environ["LORA_ALPHA"])
    return p


def detect_profile() -> tuple[str, dict]:
    """Auto-detect GPU and return (profile_name, profile_dict).

    Unknown GPU is a FATAL error — silent fallback to `safe_16gb` would train a
    very different model than the one that was validated. Use GPU_PROFILE=... to
    force a specific profile when running on unrecognized hardware.

    Per-knob env overrides (IMAGE_MAX_EDGE / LORA_R / LORA_ALPHA) are applied to
    the chosen profile via _apply_profile_overrides.
    """
    override = os.environ.get("GPU_PROFILE", "auto")
    if override != "auto":
        if override not in PROFILES:
            raise ValueError(
                f"Unknown GPU_PROFILE={override!r}. Known profiles: {sorted(PROFILES)}"
            )
        return override, _apply_profile_overrides(override, PROFILES[override])
    name = torch.cuda.get_device_name(0).lower()
    # Longer / more specific patterns first so "pro 6000" wins over any future
    # "6000" match. Order matters.
    patterns = [
        ("rtx_pro_6000_96gb", "rtx pro 6000"),
        ("rtx_pro_6000_96gb", "pro 6000"),
        ("rtx_5090_32gb",     "5090"),
        ("rtx_4090_24gb",     "4090"),
        ("rtx_3090_24gb",     "3090"),
        ("a100_80gb",         "a100"),
        ("h100_80gb",         "h100"),
        ("h200_141gb",        "h200"),
    ]
    for key, pattern in patterns:
        if pattern in name:
            return key, _apply_profile_overrides(key, PROFILES[key])
    raise RuntimeError(
        f"Unknown GPU {torch.cuda.get_device_name(0)!r}. "
        f"Set GPU_PROFILE=<name> explicitly. Known profiles: {sorted(PROFILES)}. "
        f"Silent fallback to `safe_16gb` is disabled — it would train a different "
        f"model than your validated config."
    )


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

# One source of truth for seed. Propagated to random / numpy / torch / cuda
# at train.py entry via seed_everything().
SEED           = int(os.environ.get("SEED", "3407"))
NUM_EPOCHS     = int(os.environ.get("NUM_EPOCHS", "2"))
# MAX_STEPS is a DEBUG knob. If NUM_EPOCHS is explicitly set, MAX_STEPS is
# ignored and a log line records the ignore. Prefer NUM_EPOCHS + SMOKE_TEST
# for test runs. Never set both in production.
MAX_STEPS      = int(os.environ.get("MAX_STEPS", "-1"))
REPORT_TO      = os.environ.get("REPORT_TO", "none")
SPLIT          = os.environ.get("SPLIT_STRATEGY", "splits_per_city")
SMOKE_TEST     = os.environ.get("SMOKE_TEST", "0") == "1"
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "eollm-vision-sft")
EARLY_STOPPING = os.environ.get("EARLY_STOPPING", "0") == "1"
# TEXT_ONLY=1 strips images from every record (no PIL load, no image content
# parts in the chat). Used as a non-vision baseline: how much of the dataset
# is solvable from question + options text alone? Pair with the same seed,
# epochs, and split as the vision run for an apples-to-apples comparison.
TEXT_ONLY      = os.environ.get("TEXT_ONLY", "0") == "1"

# True iff the user explicitly set NUM_EPOCHS. Used to decide whether MAX_STEPS
# should be honored as a debug cap or ignored in favor of epoch-driven training.
NUM_EPOCHS_EXPLICIT = "NUM_EPOCHS" in os.environ

_SYSTEM_PROMPT_VISION = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Answer the multiple-choice question based on the provided images. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)
# Text-only baseline removes the "based on the provided images" framing so the
# model isn't asked to reason over inputs it doesn't receive — that mismatch
# would depress accuracy for reasons unrelated to the dataset's text-only
# solvability (which is what we're measuring).
_SYSTEM_PROMPT_TEXT_ONLY = (
    "You are an urban geography expert. "
    "Answer the multiple-choice question based on the question and options text. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEXT_ONLY if TEXT_ONLY else _SYSTEM_PROMPT_VISION


def seed_everything(seed: int) -> None:
    """Seed random, numpy, torch, and CUDA in one shot.

    Called from train.py entry so all downstream randomness (dataloader
    sampling, dropout, LoRA init, generate) is seeded from a single source.
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------


def find_dataset_dir() -> Path:
    """Find EODATA_compressed_final directory."""
    explicit = os.environ.get("DATASET_DIR")
    if explicit:
        return Path(explicit)
    candidates = [
        PROJECT_ROOT / "dataset_content" / "EODATA_compressed_final",
        Path.home() / "EODATA_compressed_final",
        Path("/mnt/hdd/EODATA_compressed_final"),
    ]
    for c in candidates:
        if (c / SPLIT / "train.jsonl").exists():
            return c
    raise FileNotFoundError(
        f"Cannot find EODATA_compressed_final with {SPLIT}/train.jsonl. "
        f"Set DATASET_DIR env var. Searched: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


def generate_run_dir(profile_name: str) -> Path:
    """Create a timestamped run directory under training/runs/."""
    override = os.environ.get("OUTPUT_DIR")
    if override:
        run_dir = Path(override)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = SCRIPT_DIR / "runs" / f"{ts}_{profile_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
