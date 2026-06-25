"""
test_bs.py — standalone worst-case VRAM test at a specific batch size.

Usage:
    python training/test_bs.py 12      # test bs=12
    python training/test_bs.py 10 14   # test bs=10, then 12, then 14

What it does:
    1. Loads the same model + LoRA config as train.py
    2. Picks the K=8 worst real records (by true token count) + a synthetic
       5-image worst-case sample
    3. Runs 1 warmup + 3 scored fwd+bwd+optimizer.step iterations at each bs
    4. Reports peak reserved VRAM per iteration
    5. Exits OK if all iters fit; prints OOM + diagnostics otherwise

No training, no eval, no checkpoints. Just the VRAM stress test.
"""

from __future__ import annotations

import gc
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401 — must be first
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from config import (
    SEED, BASE_MODEL as CFG_BASE_MODEL,
    detect_profile, find_dataset_dir, seed_everything,
)
from data import (
    load_jsonl,
    select_probe_samples,
)


def _reset() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


def run_one(collator, model, optimizer, samples, bs):
    batch_records = [samples[i % len(samples)] for i in range(bs)]
    batch = collator(batch_records)
    batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def test_bs(collator, model, optimizer, samples, bs, repeats=3):
    print(f"\n=== Testing bs={bs} ===")
    _reset()
    free_before = torch.cuda.mem_get_info()[0] / 1024**2
    total = torch.cuda.mem_get_info()[1] / 1024**2
    print(f"  Free before: {free_before:.0f} MB / {total:.0f} MB total")

    # Warmup
    try:
        t0 = time.time()
        run_one(collator, model, optimizer, samples, bs)
        print(f"  Warmup: OK ({time.time()-t0:.1f}s)")
    except torch.cuda.OutOfMemoryError as e:
        _reset()
        print(f"  Warmup: OOM")
        return False

    peaks = []
    for i in range(repeats):
        _reset()
        try:
            t0 = time.time()
            run_one(collator, model, optimizer, samples, bs)
            peak_r = torch.cuda.max_memory_reserved() / 1024**2
            peak_a = torch.cuda.max_memory_allocated() / 1024**2
            peaks.append(peak_r)
            free_after = torch.cuda.mem_get_info()[0] / 1024**2
            pct_card = peak_r / total * 100
            print(f"  Iter {i+1}: OK peak_res={peak_r:.0f} MB  peak_alloc={peak_a:.0f} MB  "
                  f"({pct_card:.0f}% of card, {time.time()-t0:.1f}s, free_after={free_after:.0f} MB)")
        except torch.cuda.OutOfMemoryError:
            _reset()
            print(f"  Iter {i+1}: OOM")
            return False

    worst = max(peaks)
    print(f"  VERDICT bs={bs}: OK — worst peak_reserved={worst:.0f} MB ({worst/total*100:.0f}% of card)")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bs_list = [int(x) for x in sys.argv[1:]]

    seed_everything(SEED)
    profile_name, CFG = detect_profile()
    print(f"Profile: {profile_name}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Image max_edge: {CFG['image_max_edge']}")
    print(f"LoRA r/alpha:   {CFG['lora_r']}/{CFG['lora_alpha']}")
    print(f"Testing batch sizes: {bs_list}")
    print()

    # Load dataset
    dataset_dir = find_dataset_dir()
    split_dir = dataset_dir / "splits_per_city"
    train_records = load_jsonl(str(split_dir / "train.jsonl"))
    print(f"Dataset: {len(train_records)} train records")

    # Load model
    print("Loading Qwen3.5-4B...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CFG_BASE_MODEL,
        load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8192,
    )

    # Attach LoRA (same config as training)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=CFG["finetune_vision_layers"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=0, bias="none",
        target_modules="all-linear", random_state=SEED,
    )
    FastVisionModel.for_training(model)
    print("Model + LoRA ready.\n")

    # Build probe samples: K=8 real + synthetic (same as train.py)
    include_synth = os.environ.get("TEST_INCLUDE_SYNTHETIC", "1") == "1"
    k_real = int(os.environ.get("TEST_K_REAL", "8"))
    print(f"Selecting {k_real} real + {'1' if include_synth else '0'} synthetic worst-case samples...")
    probe_samples, info = select_probe_samples(
        train_records, base_dir=str(split_dir), max_edge=CFG["image_max_edge"],
        processor=tokenizer, k_real=k_real, include_synthetic=include_synth,
    )
    print(f"  Token lengths of real samples: {info['token_lengths']}")
    if info["synthetic_included"]:
        print(f"  Synthetic token length: {info['synthetic_token_length']}")
    print()

    # Build collator and dummy optimizer
    from unsloth.trainer import UnslothVisionDataCollator
    collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        force_match=True, completion_only_loss=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-9,
    )

    results = {}
    for bs in bs_list:
        ok = test_bs(collator, model, optimizer, probe_samples, bs)
        results[bs] = ok
        if not ok:
            print(f"\nbs={bs} FAILED — stopping higher-bs tests")
            break

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for bs in bs_list:
        status = "OK" if results.get(bs) else ("FAIL" if bs in results else "not tested")
        print(f"  bs={bs}: {status}")


if __name__ == "__main__":
    main()
