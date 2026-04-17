"""
train.py — Vision SFT on Qwen3.5-4B for EOLLM urban VQA dataset.

Usage:
    conda activate unsloth
    python train.py

Env overrides:
    GPU_PROFILE    — force a profile name (default: auto-detect)
    NUM_EPOCHS     — number of training epochs (default: 2)
    MAX_STEPS      — if set, overrides NUM_EPOCHS
    LEARNING_RATE  — override LR from profile
    SEED           — random seed (default: 3407)
    OUTPUT_DIR     — checkpoint directory (default: outputs_qwen35_4b_vision)
    REPORT_TO      — "none" or "wandb" (default: none)
    SPLIT_STRATEGY — "splits_per_city" or "splits_seen_unseen" (default: splits_per_city)
    DATASET_DIR    — path to EODATA_compressed_final (auto-detected)
    SMOKE_TEST     — set to "1" to run 5 steps only
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# GPU profiles — dataset-independent knobs only
# ---------------------------------------------------------------------------

GPU_PROFILE = os.environ.get("GPU_PROFILE", "auto")

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
    if GPU_PROFILE != "auto":
        return GPU_PROFILE, PROFILES[GPU_PROFILE]
    name = torch.cuda.get_device_name(0).lower()
    if "5090" in name:
        return "rtx_5090_32gb", PROFILES["rtx_5090_32gb"]
    elif "4090" in name:
        return "rtx_4090_24gb", PROFILES["rtx_4090_24gb"]
    elif "3090" in name:
        return "rtx_3090_24gb", PROFILES["rtx_3090_24gb"]
    elif "a100" in name:
        return "a100_80gb", PROFILES["a100_80gb"]
    elif "h100" in name:
        return "h100_80gb", PROFILES["h100_80gb"]
    elif "h200" in name:
        return "h200_141gb", PROFILES["h200_141gb"]
    else:
        print(f"WARNING: Unknown GPU '{name}', using safe_16gb profile")
        return "safe_16gb", PROFILES["safe_16gb"]


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

SEED         = int(os.environ.get("SEED", "3407"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "2"))
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "-1"))
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "outputs_qwen35_4b_vision")
REPORT_TO    = os.environ.get("REPORT_TO", "none")
SPLIT        = os.environ.get("SPLIT_STRATEGY", "splits_per_city")
SMOKE_TEST   = os.environ.get("SMOKE_TEST", "0") == "1"

# ---------------------------------------------------------------------------
# Locate dataset
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def find_dataset_dir() -> Path:
    """Find EODATA_compressed_final directory."""
    explicit = os.environ.get("DATASET_DIR")
    if explicit:
        return Path(explicit)
    candidates = [
        _PROJECT_ROOT / "dataset_content" / "EODATA_compressed_final",
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
# Dataset loading + conversion
# ---------------------------------------------------------------------------

# Add composite_utils to path
sys.path.insert(0, str(find_dataset_dir()))
from composite_utils import get_images_for_question, make_sat_marked


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resize_image(img: Image.Image, max_edge: int) -> Image.Image:
    """Resize image so longest edge <= max_edge. Convert to RGB."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


SYSTEM_PROMPT = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Answer the multiple-choice question based on the provided images. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)


def convert_record(record: dict, base_dir: str, max_edge: int) -> dict:
    """Convert one JSONL record into the Unsloth/Qwen3.5 chat format.

    Returns {"messages": [...]} with typed content parts.
    Images are resized and converted to RGB in-place.
    """
    result = get_images_for_question(record, base_dir=base_dir)
    mode = record.get("image_mode", "satellite_only")

    # Format question + options
    opts = record["options"]
    text = record["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())

    user_content: list[dict] = [{"type": "text", "text": text}]

    if mode == "satellite_arrow":
        # camera_direction: query SV + 4 option satellite-with-arrow images
        # The question asks "select the satellite image whose arrow matches this street view"
        if result.get("query_sv"):
            user_content.append({"type": "image", "image": resize_image(result["query_sv"], max_edge)})
        if result.get("options"):
            for letter in ("A", "B", "C", "D"):
                if letter in result["options"]:
                    user_content.append({"type": "image", "image": resize_image(result["options"][letter], max_edge)})

    elif mode in ("streetview_composite", "streetview_binary"):
        # Mismatch binary: question references "marked satellite image" + shows SV composite.
        # We need both: satellite_marked + the SV composite.
        sat_path = os.path.join(base_dir, record["images"]["satellite"])
        sat_marked = make_sat_marked(sat_path)
        user_content.append({"type": "image", "image": resize_image(sat_marked, max_edge)})
        user_content.append({"type": "image", "image": resize_image(result["primary"], max_edge)})

    elif mode == "streetview_mega":
        # Mismatch MCQ: question references "marked satellite image" + shows mega grid.
        # Show satellite_marked + the mega composite.
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
# Dataset
# ---------------------------------------------------------------------------

class EollmDataset:
    """On-demand converting dataset for EOLLM vision SFT.

    Converts JSONL records into Unsloth/Qwen3.5 chat format on each access.
    Conversion is fast (~18ms/sample) so it doesn't bottleneck the GPU.
    Use with num_workers=0 (conversion involves PIL Image objects).
    """

    def __init__(
        self,
        records: list[dict],
        base_dir: str,
        max_edge: int,
    ):
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
    """Measure token length distribution by running the actual Qwen3.5 processor.

    Returns dict with p50, p90, p95, p99, max, recommended max_seq_length.
    """
    import random
    rng = random.Random(SEED)
    sample_indices = rng.sample(range(len(records)), min(n_samples, len(records)))

    lengths = []
    for i, idx in enumerate(sample_indices):
        rec = records[idx]
        converted = convert_record(rec, base_dir, max_edge)

        # Collect images and build chat messages for the processor
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

        # Run the actual processor to get real token counts (text + vision)
        img_input = images[0] if len(images) == 1 else images
        inputs = processor(img_input, text, add_special_tokens=False, return_tensors="pt")
        total = inputs["input_ids"].shape[1]
        lengths.append(total)

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

    # Pick max_seq_length: p99 rounded up, with generous headroom (min 8192)
    # 300 samples isn't fully representative, so we keep a safe floor
    p99 = result["p99"]
    boundaries = [2048, 4096, 8192, 16384, 32768]
    recommended = next((b for b in boundaries if b >= p99), 32768)
    recommended = max(recommended, 8192)  # safe floor
    result["recommended_max_seq_length"] = recommended

    return result


# ---------------------------------------------------------------------------
# Batch size probing
# ---------------------------------------------------------------------------

def probe_batch_size(
    model,
    tokenizer,
    train_data: list[dict],
    max_seq_length: int,
    max_batch: int = 8,
) -> dict:
    """Probe GPU to find max batch size that fits.

    Does a forward+backward pass at increasing batch sizes.
    Returns dict with results per batch size and recommended value.
    """
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(model, tokenizer)

    results = {}
    best_bs = 1

    for bs in [1, 2, 4, 8, 16]:
        if bs > max_batch:
            break
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            batch = collator(train_data[:bs])
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            outputs.loss.backward()
            model.zero_grad()

            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            results[bs] = {"status": "OK", "peak_vram_mb": round(peak_mb)}
            best_bs = bs
            print(f"  bs={bs}: OK — peak VRAM {peak_mb:.0f} MB")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            results[bs] = {"status": "OOM"}
            print(f"  bs={bs}: OOM")
            break

        except Exception as e:
            results[bs] = {"status": f"ERROR: {e}"}
            print(f"  bs={bs}: ERROR — {e}")
            break

    torch.cuda.empty_cache()
    gc.collect()

    return {"probed": results, "recommended_batch_size": best_bs}


# ---------------------------------------------------------------------------
# Label masking sanity check
# ---------------------------------------------------------------------------

def check_label_masking(collator, train_data: list[dict], tokenizer) -> bool:
    """Verify that train_on_responses_only is working correctly.

    Decodes one sample's labels and checks that only the assistant span is non-(-100).
    """
    batch = collator([train_data[0]])
    labels = batch["labels"][0]

    # Find non-(-100) positions
    non_masked = (labels != -100).nonzero(as_tuple=True)[0]
    if len(non_masked) == 0:
        print("WARNING: ALL labels are -100 — the model will learn nothing!")
        print("  → Check instruction_part / response_part boundary strings.")
        return False

    total_tokens = len(labels)
    masked_count = (labels == -100).sum().item()
    unmasked_count = len(non_masked)

    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked (instruction/padding): {masked_count}")
    print(f"  Unmasked (assistant response): {unmasked_count}")

    # Decode the unmasked part
    unmasked_ids = labels[non_masked]
    decoded = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
    print(f"  Unmasked text: {repr(decoded[:200])}")

    if unmasked_count == total_tokens:
        print("WARNING: NO labels are masked — train_on_responses_only may not be working!")
        return False

    return True


# ---------------------------------------------------------------------------
# Eval samples (base vs finetuned)
# ---------------------------------------------------------------------------

def run_eval_samples(
    model,
    tokenizer,
    records: list[dict],
    base_dir: str,
    max_edge: int,
    n: int = 3,
    tag: str = "base",
) -> list[dict]:
    """Run inference on n samples and return results."""
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    import random
    rng = random.Random(SEED + 1)  # different seed than training
    indices = rng.sample(range(len(records)), min(n, len(records)))
    results = []

    for idx in indices:
        rec = records[idx]
        converted = convert_record(rec, base_dir, max_edge)

        # Build inference messages (no assistant turn)
        user_msg = converted["messages"][1]  # user turn
        images = [part["image"] for part in user_msg["content"] if part["type"] == "image"]

        inf_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                p if p["type"] == "text" else {"type": "image"}
                for p in user_msg["content"]
            ]},
        ]

        text = tokenizer.apply_chat_template(inf_messages, add_generation_prompt=True, tokenize=False)
        img_input = images[0] if len(images) == 1 else images
        inputs = tokenizer(img_input, text, add_special_tokens=False, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                use_cache=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        results.append({
            "question_id": rec["question_id"],
            "topic": rec["topic"],
            "question": rec["question"],
            "gold": rec["answer"],
            "predicted": generated,
            "correct": generated.startswith(rec["answer"]),
        })

    return results


def write_eval_md(base_results: list[dict], ft_results: list[dict], path: str):
    """Write eval_samples.md comparing base vs finetuned."""
    lines = ["# Eval Samples — Base vs Finetuned\n"]
    for b, f in zip(base_results, ft_results):
        lines.append(f"## {b['question_id']} ({b['topic']})\n")
        lines.append(f"**Question:** {b['question']}\n")
        lines.append(f"**Gold answer:** {b['gold']}\n")
        lines.append(f"| Model | Predicted | Correct |")
        lines.append(f"|-------|-----------|---------|")
        lines.append(f"| Base | {b['predicted']} | {'yes' if b['correct'] else 'no'} |")
        lines.append(f"| Finetuned | {f['predicted']} | {'yes' if f['correct'] else 'no'} |")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # --- Profile ---
    profile_name, CFG = detect_profile()
    lr = float(os.environ.get("LEARNING_RATE", str(CFG["lr"])))

    print("=" * 60)
    print("EOLLM Vision SFT — Qwen3.5-4B")
    print("=" * 60)
    print(f"GPU profile:      {profile_name}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Image max edge:   {CFG['image_max_edge']} px")
    print(f"LoRA rank:        {CFG['lora_r']}")
    print(f"Learning rate:    {lr}")
    print(f"Vision layers:    {CFG['finetune_vision_layers']}")
    print(f"Split strategy:   {SPLIT}")
    print(f"Seed:             {SEED}")
    print(f"Smoke test:       {SMOKE_TEST}")
    print()

    # --- Locate dataset ---
    dataset_dir = find_dataset_dir()
    split_dir = dataset_dir / SPLIT
    print(f"Dataset dir:      {dataset_dir}")
    print(f"Split dir:        {split_dir}")

    train_jsonl = str(split_dir / "train.jsonl")
    val_jsonl = str(split_dir / "validation.jsonl")

    print("Loading JSONL records...")
    train_records = load_jsonl(train_jsonl)
    val_records = load_jsonl(val_jsonl)
    print(f"  Train: {len(train_records)} records")
    print(f"  Val:   {len(val_records)} records")
    print()

    # --- Load model ---
    print("Loading Qwen3.5-4B (bf16 LoRA)...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=4096,  # initial; will be adjusted after measurement
    )
    print("Model loaded.")
    print()

    # --- Measure token lengths ---
    print("Measuring token length distribution (300 samples)...")
    token_stats = measure_token_lengths(
        train_records, str(split_dir), CFG["image_max_edge"], tokenizer, n_samples=300
    )
    max_seq_length = token_stats["recommended_max_seq_length"]

    print(f"\nToken length distribution:")
    print(f"  min:  {token_stats['min']}")
    print(f"  p50:  {token_stats['p50']}")
    print(f"  p90:  {token_stats['p90']}")
    print(f"  p95:  {token_stats['p95']}")
    print(f"  p99:  {token_stats['p99']}")
    print(f"  max:  {token_stats['max']}")
    print(f"  → max_seq_length: {max_seq_length}")
    print()

    # Write token budget report
    token_budget_path = _SCRIPT_DIR / "token_budget.md"
    with open(token_budget_path, "w") as f:
        f.write("# Token Budget Report\n\n")
        f.write(f"Profile: {profile_name}\n")
        f.write(f"Image max edge: {CFG['image_max_edge']} px\n")
        f.write(f"Measured on: {token_stats['n_samples']} samples\n\n")
        f.write("## Distribution\n\n")
        f.write(f"| Percentile | Tokens |\n")
        f.write(f"|------------|--------|\n")
        for k in ["min", "p50", "p90", "p95", "p99", "max"]:
            f.write(f"| {k} | {token_stats[k]} |\n")
        f.write(f"\n**Selected max_seq_length: {max_seq_length}**\n")
        f.write(f"\nRationale: covers p99 ({token_stats['p99']}) rounded to next clean boundary.\n")
    print(f"Token budget written to {token_budget_path}")

    # If p99 is suspiciously small or huge, warn
    if token_stats["p99"] < 256:
        print(f"NOTE: p99 is only {token_stats['p99']} tokens — sequences are very short.")
        print("      Qwen3.5-4B might be overkill for this task.")
    elif token_stats["p99"] > 16384:
        print(f"WARNING: p99 is {token_stats['p99']} tokens — very long sequences!")
        print("         Consider downscaling images or truncating.")

    # --- Attach LoRA ---
    print("Attaching LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=CFG["finetune_vision_layers"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        target_modules="all-linear",
        random_state=SEED,
    )
    print("LoRA attached.")
    print()

    # --- Convert a small batch for probing ---
    print("Converting probe samples...")
    probe_samples = [
        convert_record(train_records[i], str(split_dir), CFG["image_max_edge"])
        for i in range(min(32, len(train_records)))
    ]

    # --- Batch size probing ---
    print("Probing batch size...")
    FastVisionModel.for_training(model)
    probe_result = probe_batch_size(
        model, tokenizer, probe_samples, max_seq_length,
        max_batch=CFG["initial_batch_guess"] * 2,
    )
    batch_size = probe_result["recommended_batch_size"]

    # Compute gradient accumulation to hit effective batch of 8-32
    target_effective = max(8, min(32, len(train_records) // 1000))
    grad_accum = max(1, target_effective // batch_size)
    effective_batch = batch_size * grad_accum
    step_size = effective_batch

    print(f"\n  Batch size:       {batch_size}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch:  {effective_batch}")
    print()

    # Write tuning notes
    tuning_path = _SCRIPT_DIR / "tuning_notes.md"
    with open(tuning_path, "w") as f:
        f.write("# Tuning Notes — Batch Probe Results\n\n")
        f.write(f"Profile: {profile_name}\n")
        f.write(f"max_seq_length: {max_seq_length}\n\n")
        f.write("## Batch probe\n\n")
        f.write("| Batch size | Status | Peak VRAM (MB) |\n")
        f.write("|------------|--------|----------------|\n")
        for bs, info in probe_result["probed"].items():
            vram = info.get("peak_vram_mb", "—")
            f.write(f"| {bs} | {info['status']} | {vram} |\n")
        f.write(f"\n**Selected batch_size: {batch_size}**\n")
        f.write(f"**Gradient accumulation: {grad_accum}**\n")
        f.write(f"**Effective batch: {effective_batch}**\n")
        f.write(f"\nVRAM budget: {CFG['vram_gb']} GB\n")
        peak_at_best = probe_result["probed"].get(batch_size, {}).get("peak_vram_mb", 0)
        if peak_at_best:
            pct = peak_at_best / (CFG["vram_gb"] * 1024) * 100
            f.write(f"Peak at bs={batch_size}: {peak_at_best} MB ({pct:.0f}% of budget)\n")
    print(f"Tuning notes written to {tuning_path}")

    # --- Label masking check ---
    print("\nChecking label masking...")
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        force_match=True,
        completion_only_loss=True,
    )
    masking_ok = check_label_masking(collator, probe_samples, tokenizer)
    if not masking_ok:
        print("FATAL: Label masking check failed. Aborting.")
        sys.exit(1)
    print("Label masking OK.\n")

    # --- Eval samples (base model) ---
    print("Running eval on base model (3 samples)...")
    base_eval = run_eval_samples(
        model, tokenizer, val_records, str(split_dir), CFG["image_max_edge"], n=3, tag="base"
    )
    for r in base_eval:
        print(f"  {r['question_id']}: gold={r['gold']}, pred={r['predicted']}, correct={r['correct']}")
    print()

    # --- Build dataset ---
    print("Building training dataset...")
    train_dataset = EollmDataset(
        records=train_records,
        base_dir=str(split_dir),
        max_edge=CFG["image_max_edge"],
    )
    print()

    # --- Training ---
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    steps_per_epoch = math.ceil(len(train_records) / effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = min(40, total_steps // 10)
    save_steps = max(100, steps_per_epoch // 2)

    if SMOKE_TEST:
        MAX_STEPS_FINAL = 5
        save_steps = 5
    elif MAX_STEPS > 0:
        MAX_STEPS_FINAL = MAX_STEPS
    else:
        MAX_STEPS_FINAL = -1  # use num_train_epochs

    print(f"Training config:")
    print(f"  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total epochs:     {NUM_EPOCHS}")
    print(f"  Total steps:      {total_steps if MAX_STEPS_FINAL < 0 else MAX_STEPS_FINAL}")
    print(f"  Warmup steps:     {warmup_steps}")
    print(f"  Save steps:       {save_steps}")
    print(f"  max_seq_length:   {max_seq_length}")
    print()

    # --- Build validation dataset ---
    val_dataset = EollmDataset(
        records=val_records,
        base_dir=str(split_dir),
        max_edge=CFG["image_max_edge"],
    )

    eval_steps = max(50, steps_per_epoch // 4)  # ~4 evals per epoch

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            # Mandatory vision flags
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,

            # Training knobs
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=NUM_EPOCHS if MAX_STEPS_FINAL < 0 else 1,
            max_steps=MAX_STEPS_FINAL if MAX_STEPS_FINAL > 0 else -1,
            warmup_steps=warmup_steps,
            learning_rate=lr,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,

            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=3,
            seed=SEED,
            bf16=True,
            output_dir=OUTPUT_DIR,
            report_to=REPORT_TO,
            dataloader_num_workers=0,  # PIL Images can't be pickled across workers

            # Validation
            eval_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=batch_size,
        ),
    )

    # Add a callback to log loss to CSV for plotting
    from transformers import TrainerCallback

    loss_log_path = str(_SCRIPT_DIR / "training_loss.csv")

    class LossLogger(TrainerCallback):
        def __init__(self):
            self.steps = []
            self.losses = []
            self.lrs = []
            self.eval_steps = []
            self.eval_losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.steps.append(state.global_step)
                self.losses.append(logs["loss"])
                self.lrs.append(logs.get("learning_rate", 0))
                # Write train CSV incrementally
                with open(loss_log_path, "w") as f:
                    f.write("step,loss,learning_rate\n")
                    for s, l, r in zip(self.steps, self.losses, self.lrs):
                        f.write(f"{s},{l},{r}\n")
            if logs and "eval_loss" in logs:
                self.eval_steps.append(state.global_step)
                self.eval_losses.append(logs["eval_loss"])
                # Write eval CSV incrementally
                eval_csv_path = loss_log_path.replace("training_loss", "eval_loss")
                with open(eval_csv_path, "w") as f:
                    f.write("step,eval_loss\n")
                    for s, l in zip(self.eval_steps, self.eval_losses):
                        f.write(f"{s},{l}\n")

    loss_logger = LossLogger()
    trainer.add_callback(loss_logger)

    print("Starting training...")
    if SMOKE_TEST:
        print("  (SMOKE TEST — 5 steps only)")
    print()

    trainer_stats = trainer.train()


    # --- Generate training plots ---
    if loss_logger.steps:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].plot(loss_logger.steps, loss_logger.losses, "b-", linewidth=1.5, label="Train")
            if loss_logger.eval_steps:
                axes[0].plot(loss_logger.eval_steps, loss_logger.eval_losses, "ro-",
                             markersize=6, linewidth=2, label="Validation")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training & Validation Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(loss_logger.steps, loss_logger.lrs, "r-", linewidth=1.5)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_title("Learning Rate Schedule")
            axes[1].grid(True, alpha=0.3)

            # Loss histogram (last 50% of training)
            mid = len(loss_logger.losses) // 2
            if mid > 0:
                axes[2].hist(loss_logger.losses[mid:], bins=30, color="steelblue", alpha=0.8)
                axes[2].set_xlabel("Loss")
                axes[2].set_ylabel("Count")
                axes[2].set_title(f"Loss Distribution (steps {loss_logger.steps[mid]}–{loss_logger.steps[-1]})")
                axes[2].grid(True, alpha=0.3)

            fig.suptitle(f"EOLLM Vision SFT — Qwen3.5-4B ({profile_name})", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plot_path = str(_SCRIPT_DIR / "training_curves.png")
            fig.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Training curves saved to {plot_path}")
        except ImportError:
            print("matplotlib not available — skipping plots")

    # --- Save ---
    lora_dir = str(_SCRIPT_DIR / "qwen35-4b-vision-lora")
    print(f"\nSaving LoRA adapter to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    if not SMOKE_TEST:
        merged_dir = str(_SCRIPT_DIR / "qwen35-4b-vision-merged")
        print(f"Saving merged bf16 to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # --- Eval samples (finetuned) ---
    print("\nRunning eval on finetuned model (3 samples)...")
    ft_eval = run_eval_samples(
        model, tokenizer, val_records, str(split_dir), CFG["image_max_edge"], n=3, tag="finetuned"
    )
    for r in ft_eval:
        print(f"  {r['question_id']}: gold={r['gold']}, pred={r['predicted']}, correct={r['correct']}")

    eval_path = str(_SCRIPT_DIR / "eval_samples.md")
    write_eval_md(base_eval, ft_eval, eval_path)
    print(f"Eval comparison written to {eval_path}")

    # --- Summary ---
    t_total = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_pct = peak_vram_mb / (CFG["vram_gb"] * 1024) * 100

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Profile:           {profile_name}")
    print(f"max_seq_length:    {max_seq_length} (from p99={token_stats['p99']})")
    print(f"Batch size:        {batch_size} × {grad_accum} = {effective_batch} effective")
    print(f"Final loss:        {trainer_stats.training_loss:.4f}")
    print(f"Wall clock:        {t_total/60:.1f} min")
    print(f"Peak VRAM:         {peak_vram_mb:.0f} MB ({peak_pct:.0f}% of {CFG['vram_gb']} GB budget)")
    print(f"LoRA adapter:      {lora_dir}")
    if not SMOKE_TEST:
        print(f"Merged model:      {merged_dir}")
    print(f"Checkpoints:       {OUTPUT_DIR}")
    print(f"Eval samples:      {eval_path}")

    if peak_pct < 70:
        print(f"\nNOTE: Peak VRAM is {peak_pct:.0f}% of budget — GPU underutilized.")
        print("      Consider increasing batch_size or image_max_edge.")
    elif peak_pct > 95:
        print(f"\nNOTE: Peak VRAM is {peak_pct:.0f}% — very tight. Monitor for OOM on long sequences.")

    print("\nDone.")


if __name__ == "__main__":
    main()
