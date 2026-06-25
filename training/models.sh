#!/usr/bin/env bash
# models.sh — model registry for the multi-model training sweep.
#
# Sourced by launch_multimodel.sh. Defines one entry per model. Each entry is a
# pipe-separated tuple:
#
#   KEY | BASE_MODEL | MODEL_FAMILY | MODEL_NAME | LEARNING_RATE | NUM_EPOCHS
#
#   KEY           short slug used in run/output dir names and the eval registry
#   BASE_MODEL    HF hub id OR local path. On lab-ws these resolve from the HF
#                 cache (already downloaded). config.py reads this verbatim.
#   MODEL_FAMILY  qwen | gemma  — drives chat template + collator masking strings
#                 (Qwen <|im_start|> vs Gemma <|turn>). MUST be correct or label
#                 masking silently breaks (train.py check_label_masking guards it).
#   MODEL_NAME    display label for W&B / summary.txt
#   LEARNING_RATE per-size LR. Small models (4B/E2B) use the validated 2e-4;
#                 big models (27B/31B) halve to 1e-4 for stability at LoRA r=16.
#   NUM_EPOCHS    per-size epoch budget. Big models (27B/31B) = 12; small models
#                 (4B/E2B) = 6. EARLY_STOPPING=1 (patience=2) can stop earlier.
#
# Verified present in the lab-ws HF cache (2026-06):
#   unsloth/Qwen3.5-27B          52 GB, 11 safetensors  ✓
#   unsloth/gemma-4-31B-it       59 GB,  2 safetensors  ✓
#   unsloth/gemma-4-E2B-it      9.6 GB                  ✓
#   models/Qwen3.5-4B (local)   8.8 GB                  ✓
#
# All models share: LoRA r/alpha=16, image_max_edge=512, adamw_8bit, bf16 LoRA,
# SAVE_MERGED=0 (adapter-only), probe per model. Those are set in the launcher,
# not here — this file is ONLY the per-model identity + LR.

# Order matters: the launcher trains models top-to-bottom. BOTH Qwens first
# (user preference — validate the Qwen family fully before any Gemma run), then
# the Gemmas. Within Qwen: 4B first (its seen_unseen run is the validation gate
# before 27B commits GPU days). Within Gemma: E2B first (fast shakeout of the
# Gemma chat-template / <|turn> masking path before the 31B).
MODELS=(
    "qwen4b|${QWEN4B_BASE:-/home/ain480/training/models/Qwen3.5-4B}|qwen|Qwen3.5-4B|2e-4|6"
    "qwen27b|unsloth/Qwen3.5-27B|qwen|Qwen3.5-27B|1e-4|12"
    "gemma_e2b|unsloth/gemma-4-E2B-it|gemma|gemma-4-E2B-it|2e-4|6"
    "gemma_31b|unsloth/gemma-4-31B-it|gemma|gemma-4-31B-it|1e-4|12"
)

# The three training conditions per model. Pipe-separated:
#   COND_KEY | SPLIT_STRATEGY | TEXT_ONLY
#
#   per_city     vision SFT on splits_per_city
#   seen_unseen  vision SFT on splits_seen_unseen (cross-city generalization)
#   text_only    non-vision baseline on splits_per_city (TEXT_ONLY=1)
#
# seen_unseen is listed FIRST so the very first job in the whole sweep is
# "qwen4b seen_unseen" — the validation gate the user asked for.
CONDITIONS=(
    "seen_unseen|splits_seen_unseen|0"
    "per_city|splits_per_city|0"
    "text_only|splits_per_city|1"
)
