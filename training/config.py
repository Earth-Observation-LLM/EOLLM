"""
config.py — GPU profiles, env overrides, paths, and run directory management.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# GPU profiles — dataset-independent knobs only
# ---------------------------------------------------------------------------

PROFILES = {
    "rtx_5090_32gb": dict(vram_gb=32,  image_max_edge=768,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=8,  finetune_vision_layers=True),
    "rtx_4090_24gb": dict(vram_gb=24,  image_max_edge=640,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=2,  finetune_vision_layers=True),
    "rtx_3090_24gb": dict(vram_gb=24,  image_max_edge=512,  lora_r=16, lora_alpha=16, lr=2e-4, initial_batch_guess=2,  finetune_vision_layers=True),
    "a100_80gb":     dict(vram_gb=80,  image_max_edge=1024, lora_r=32, lora_alpha=32, lr=1e-4, initial_batch_guess=8,  finetune_vision_layers=True),
    "h100_80gb":     dict(vram_gb=80,  image_max_edge=1024, lora_r=32, lora_alpha=32, lr=1e-4, initial_batch_guess=8,  finetune_vision_layers=True),
    "h200_141gb":    dict(vram_gb=141, image_max_edge=1024, lora_r=64, lora_alpha=64, lr=1e-4, initial_batch_guess=16, finetune_vision_layers=True),
    "safe_16gb":     dict(vram_gb=16,  image_max_edge=448,  lora_r=8,  lora_alpha=8,  lr=2e-4, initial_batch_guess=1,  finetune_vision_layers=False),
}


def detect_profile() -> tuple[str, dict]:
    """Auto-detect GPU and return (profile_name, profile_dict)."""
    override = os.environ.get("GPU_PROFILE", "auto")
    if override != "auto":
        return override, PROFILES[override]
    name = torch.cuda.get_device_name(0).lower()
    for key, pattern in [
        ("rtx_5090_32gb", "5090"), ("rtx_4090_24gb", "4090"),
        ("rtx_3090_24gb", "3090"), ("a100_80gb", "a100"),
        ("h100_80gb", "h100"), ("h200_141gb", "h200"),
    ]:
        if pattern in name:
            return key, PROFILES[key]
    print(f"WARNING: Unknown GPU '{name}', using safe_16gb profile")
    return "safe_16gb", PROFILES["safe_16gb"]


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

SEED         = int(os.environ.get("SEED", "3407"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "2"))
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "-1"))
REPORT_TO    = os.environ.get("REPORT_TO", "none")
SPLIT        = os.environ.get("SPLIT_STRATEGY", "splits_per_city")
SMOKE_TEST   = os.environ.get("SMOKE_TEST", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "eollm-vision-sft")

SYSTEM_PROMPT = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Answer the multiple-choice question based on the provided images. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)

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
