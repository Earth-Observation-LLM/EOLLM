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


def detect_profile() -> tuple[str, dict]:
    """Auto-detect GPU and return (profile_name, profile_dict).

    Unknown GPU is a FATAL error — silent fallback to `safe_16gb` would train a
    very different model than the one that was validated. Use GPU_PROFILE=... to
    force a specific profile when running on unrecognized hardware.
    """
    override = os.environ.get("GPU_PROFILE", "auto")
    if override != "auto":
        if override not in PROFILES:
            raise ValueError(
                f"Unknown GPU_PROFILE={override!r}. Known profiles: {sorted(PROFILES)}"
            )
        return override, PROFILES[override]
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
            return key, PROFILES[key]
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

# True iff the user explicitly set NUM_EPOCHS. Used to decide whether MAX_STEPS
# should be honored as a debug cap or ignored in favor of epoch-driven training.
NUM_EPOCHS_EXPLICIT = "NUM_EPOCHS" in os.environ

SYSTEM_PROMPT = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Answer the multiple-choice question based on the provided images. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)


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
