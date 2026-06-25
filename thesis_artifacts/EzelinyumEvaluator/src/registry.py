"""Model registry — only the 3 runs for the panosu."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    key: str
    display: str
    backend: str                # "unsloth" — Qwen3.5-4B uses unsloth
    hf_id: str
    lora_path: Optional[str] = None
    family: str = "baseline"
    notes: str = ""
    enable_thinking: bool = False  # all 3 runs have thinking OFF
    max_new_tokens: int = 8


QWEN35_BASE = "/home/ain480/training/models/Qwen3.5-4B"
RUNS = "/home/ain480/training/training/runs"


REGISTRY: dict[str, ModelSpec] = {
    "qwen3_5_4b_base": ModelSpec(
        key="qwen3_5_4b_base",
        display="Qwen3.5-4B base (untrained, thinking OFF)",
        backend="unsloth",
        hf_id=QWEN35_BASE,
        family="baseline",
    ),
    "su_base": ModelSpec(
        key="su_base",
        display="SU-base — seen_unseen LoRA",
        backend="unsloth",
        hf_id=QWEN35_BASE,
        lora_path=f"{RUNS}/20260423_190939_rtx_pro_6000_96gb/lora",
        family="finetuned",
    ),
    "pc_base": ModelSpec(
        key="pc_base",
        display="PC-base — per_city LoRA (full 14 topics, 6 epochs)",
        backend="unsloth",
        hf_id=QWEN35_BASE,
        lora_path=f"{RUNS}/20260422_074420_rtx_pro_6000_96gb/checkpoints/checkpoint-2022",
        family="finetuned",
        notes="The REAL PC-base flagship — 78.3% on 6984-sample per_city val (report.tex headline).",
    ),
}


def get(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]


def all_keys() -> list[str]:
    return list(REGISTRY.keys())
