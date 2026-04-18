"""
train.py — Vision SFT on Qwen3.5-4B for EOLLM urban VQA.

Usage:
    conda activate unsloth
    python train.py                          # full run, no W&B
    REPORT_TO=wandb python train.py          # full run with W&B
    SMOKE_TEST=1 python train.py             # 5 steps, quick validation
    NUM_EPOCHS=3 LEARNING_RATE=1e-4 python train.py
    EARLY_STOPPING=1 python train.py         # stop if val acc drops for 2 evals

Stop conditions (IMPORTANT — do not set both):
    - NUM_EPOCHS is authoritative. If explicitly set, MAX_STEPS is ignored.
    - MAX_STEPS is a DEBUG knob. Use it only for quick partial runs (and then
      only if NUM_EPOCHS is not set). Setting both will log that MAX_STEPS is
      being ignored and continue with NUM_EPOCHS.

All outputs go to training/runs/{timestamp}_{profile}/
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Unsloth must be imported before torch/transformers so its patching is applied.
# The warning "Unsloth should be imported first" otherwise floods the log.
import unsloth  # noqa: F401

import torch

from config import (
    SEED, NUM_EPOCHS, MAX_STEPS, REPORT_TO, SPLIT, SMOKE_TEST,
    WANDB_PROJECT, SCRIPT_DIR, BASE_MODEL as CFG_BASE_MODEL,
    EARLY_STOPPING, NUM_EPOCHS_EXPLICIT,
    detect_profile, find_dataset_dir, generate_run_dir, seed_everything,
)
from data import load_jsonl, convert_record, measure_token_lengths, EollmDataset
from evaluation import run_eval_samples, write_eval_md, compute_topic_accuracy, write_accuracy_md
from callbacks import LossLogger, EvalCallback, plot_training_curves


# ---------------------------------------------------------------------------
# Batch size probing
# ---------------------------------------------------------------------------


# Probe list extended past 16 for the 96 GB profile. Add a safety margin by
# taking one size-tier below the largest successful bs (see main()).
_PROBE_SIZES = [1, 2, 4, 8, 16, 24, 32]


def probe_batch_size(model, tokenizer, train_data: list[dict], max_batch: int = 16) -> dict:
    """Probe GPU to find max batch size that fits."""
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(model, tokenizer)
    results = {}
    best_bs = 1

    for bs in _PROBE_SIZES:
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


def apply_safety_margin(raw_bs: int, probed: dict) -> int:
    """Step one probe tier below the largest OK bs to absorb worst-case activation
    spikes during real training (optimizer state + gradient checkpointing peaks)."""
    if raw_bs <= 1:
        return 1
    ok_sizes = sorted(bs for bs, info in probed.items() if info.get("status") == "OK")
    if len(ok_sizes) < 2:
        return raw_bs
    # Drop one tier from the top
    return ok_sizes[-2]


def check_label_masking(collator, sample: dict, tokenizer) -> bool:
    """Verify train_on_responses_only masks correctly."""
    batch = collator([sample])
    labels = batch["labels"][0]

    non_masked = (labels != -100).nonzero(as_tuple=True)[0]
    if len(non_masked) == 0:
        print("FATAL: ALL labels are -100!")
        return False

    total = len(labels)
    unmasked = len(non_masked)
    decoded = tokenizer.decode(labels[non_masked], skip_special_tokens=False)
    print(f"  Total: {total}, masked: {total - unmasked}, unmasked: {unmasked}")
    print(f"  Unmasked text: {repr(decoded[:200])}")

    if unmasked == total:
        print("FATAL: NO labels masked!")
        return False
    return True


def _safe_wandb_log(payload: dict, step: int) -> None:
    """wandb.log wrapper that never propagates exceptions."""
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(payload, step=step)
    except Exception as e:
        print(f"[wandb] log failed (continuing): {e}")


# ---------------------------------------------------------------------------
# Stop condition resolution
# ---------------------------------------------------------------------------


def resolve_stop_condition() -> tuple[int, int]:
    """Resolve (num_train_epochs, max_steps) from env flags.

    Rules:
      - SMOKE_TEST overrides everything to (1, 5).
      - If NUM_EPOCHS is explicitly set, MAX_STEPS is ignored (log the ignore).
      - If only MAX_STEPS is set (>0), use it with num_train_epochs=1.
      - Otherwise default to NUM_EPOCHS epochs, no max_steps cap.

    Returns (num_train_epochs_arg, max_steps_arg) as expected by SFTConfig.
    """
    if SMOKE_TEST:
        print("[stop] SMOKE_TEST=1 → max_steps=5, num_train_epochs=1")
        return 1, 5

    if NUM_EPOCHS_EXPLICIT:
        if MAX_STEPS > 0:
            print(f"[stop] NUM_EPOCHS={NUM_EPOCHS} explicitly set — IGNORING MAX_STEPS={MAX_STEPS}.")
            print("[stop] Do not set both. NUM_EPOCHS wins.")
        else:
            print(f"[stop] NUM_EPOCHS={NUM_EPOCHS}, no max_steps cap.")
        return NUM_EPOCHS, -1

    if MAX_STEPS > 0:
        print(f"[stop] MAX_STEPS={MAX_STEPS} (debug cap), num_train_epochs=1.")
        return 1, MAX_STEPS

    print(f"[stop] NUM_EPOCHS={NUM_EPOCHS} (default), no max_steps cap.")
    return NUM_EPOCHS, -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.time()

    # One source of truth for all randomness (random, numpy, torch, cuda).
    seed_everything(SEED)

    # --- Profile + run dir ---
    profile_name, CFG = detect_profile()
    lr = float(os.environ.get("LEARNING_RATE", str(CFG["lr"])))
    run_dir = generate_run_dir(profile_name)

    print("=" * 60)
    print("EOLLM Vision SFT — Qwen3.5-4B")
    print("=" * 60)
    print(f"Profile:          {profile_name}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Image max edge:   {CFG['image_max_edge']} px")
    print(f"LoRA r/alpha:     {CFG['lora_r']}/{CFG['lora_alpha']}")
    print(f"LR:               {lr}")
    print(f"Vision layers:    {CFG['finetune_vision_layers']}")
    print(f"Split:            {SPLIT}")
    print(f"Seed:             {SEED}")
    print(f"Epochs:           {NUM_EPOCHS}")
    print(f"Smoke test:       {SMOKE_TEST}")
    print(f"Early stopping:   {EARLY_STOPPING}")
    print(f"Report to:        {REPORT_TO}")
    print(f"Run dir:          {run_dir}")
    print()

    # --- W&B init ---
    use_wandb = REPORT_TO == "wandb"
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                name=run_dir.name,
                config={
                    "model": "Qwen3.5-4B", "method": "bf16 LoRA",
                    "profile": profile_name, "image_max_edge": CFG["image_max_edge"],
                    "lora_r": CFG["lora_r"], "lora_alpha": CFG["lora_alpha"],
                    "lr": lr, "finetune_vision_layers": CFG["finetune_vision_layers"],
                    "split": SPLIT, "seed": SEED, "num_epochs": NUM_EPOCHS,
                    "early_stopping": EARLY_STOPPING,
                },
                dir=str(run_dir),
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")
            print("[wandb] Continuing with REPORT_TO=none. Set WANDB_API_KEY or run `wandb login`.")
            use_wandb = False

    # --- Load data ---
    dataset_dir = find_dataset_dir()
    split_dir = dataset_dir / SPLIT
    print(f"Dataset:          {split_dir}")
    train_records = load_jsonl(str(split_dir / "train.jsonl"))
    val_records = load_jsonl(str(split_dir / "validation.jsonl"))
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}\n")

    # --- Load model ---
    print("Loading Qwen3.5-4B (bf16 LoRA)...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CFG_BASE_MODEL,
        load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8192,
    )
    print("Model loaded.\n")

    # --- Measure tokens ---
    print("Measuring token lengths (300 samples)...")
    token_stats = measure_token_lengths(
        train_records, str(split_dir), CFG["image_max_edge"], tokenizer, n_samples=300
    )
    max_seq_length = token_stats["recommended_max_seq_length"]
    print(f"  p50={token_stats['p50']}  p95={token_stats['p95']}  p99={token_stats['p99']}  max={token_stats['max']}")
    print(f"  -> max_seq_length: {max_seq_length}\n")

    with open(run_dir / "token_budget.md", "w") as f:
        f.write("# Token Budget\n\n")
        f.write(f"Profile: {profile_name}, image_max_edge: {CFG['image_max_edge']}px\n\n")
        f.write("| Stat | Tokens |\n|------|--------|\n")
        for k in ["min", "p50", "p90", "p95", "p99", "max"]:
            f.write(f"| {k} | {token_stats[k]} |\n")
        f.write(f"\n**max_seq_length: {max_seq_length}**\n")

    # --- Attach LoRA ---
    print("Attaching LoRA...")
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
    print("LoRA attached.\n")

    # --- Probe samples — worst-case batches for accurate VRAM estimation ---
    print("Building probe samples (worst-case: 5-image mismatch_binary)...")
    # Probe must test with the heaviest samples (5-image mismatch_binary/composite)
    # to avoid OOM on unlucky batches during training
    heavy_indices = [i for i, r in enumerate(train_records)
                     if r["image_mode"] in ("streetview_composite", "streetview_binary")][:32]
    if len(heavy_indices) < 16:
        # fallback: mix of heavy modes
        for i, r in enumerate(train_records):
            if len(heavy_indices) >= 32:
                break
            if i not in set(heavy_indices):
                heavy_indices.append(i)
    probe_samples = [convert_record(train_records[i], str(split_dir), CFG["image_max_edge"]) for i in heavy_indices]
    print(f"  Built {len(probe_samples)} probe samples (all 5-image worst-case)")

    # --- Probe batch size ---
    print("Probing batch size...")
    FastVisionModel.for_training(model)
    # Probe up to 2x the initial guess, capped by the probe list's max.
    probe_cap = min(max(_PROBE_SIZES), CFG["initial_batch_guess"] * 2)
    probe_result = probe_batch_size(model, tokenizer, probe_samples, max_batch=probe_cap)
    raw_bs = probe_result["recommended_batch_size"]
    batch_size = apply_safety_margin(raw_bs, probe_result["probed"])
    if batch_size < raw_bs:
        print(f"  safety margin: {raw_bs} -> {batch_size} (one tier down)")

    target_effective = max(8, min(32, len(train_records) // 1000))
    grad_accum = max(1, target_effective // batch_size)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_records) / effective_batch)

    print(f"\n  bs={batch_size} x grad_accum={grad_accum} = {effective_batch} effective")
    print(f"  steps/epoch: {steps_per_epoch}\n")

    with open(run_dir / "tuning_notes.md", "w") as f:
        f.write("# Tuning Notes\n\n")
        f.write(f"max_seq_length: {max_seq_length}\n\n")
        f.write("| BS | Status | Peak VRAM (MB) |\n|-----|--------|----------------|\n")
        for bs, info in probe_result["probed"].items():
            f.write(f"| {bs} | {info['status']} | {info.get('peak_vram_mb', '—')} |\n")
        f.write(f"\n**probe best bs={raw_bs}, used bs={batch_size} (safety margin), ")
        f.write(f"grad_accum={grad_accum}, effective={effective_batch}**\n")

    # --- Label masking check ---
    print("Checking label masking...")
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        force_match=True, completion_only_loss=True,
    )
    if not check_label_masking(collator, probe_samples[0], tokenizer):
        sys.exit(1)
    print("Label masking OK.\n")

    # --- Baseline eval (base model accuracy before training) ---
    print("Computing baseline accuracy (100 val samples, greedy, shared parser)...")
    baseline_accuracy = compute_topic_accuracy(
        model, tokenizer, val_records, str(split_dir), CFG["image_max_edge"],
        n=100, seed=SEED,
    )
    print(f"  Base model: {baseline_accuracy['overall']:.1%}")
    for topic, info in baseline_accuracy["per_topic"].items():
        print(f"    {topic}: {info['acc']:.0%}")

    if use_wandb:
        _safe_wandb_log({"baseline/accuracy": baseline_accuracy["overall"]}, step=0)
        for topic, info in baseline_accuracy["per_topic"].items():
            _safe_wandb_log({f"baseline/acc_{topic}": info["acc"]}, step=0)

    print()

    # --- Build datasets ---
    train_dataset = EollmDataset(train_records, str(split_dir), CFG["image_max_edge"])
    val_subset = val_records[:200] if SMOKE_TEST else val_records
    val_dataset = EollmDataset(val_subset, str(split_dir), CFG["image_max_edge"])

    # --- Training config ---
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback
    FastVisionModel.for_training(model)

    num_train_epochs_arg, max_steps_arg = resolve_stop_condition()

    # total_steps used for warmup/scheduler sizing only
    if max_steps_arg > 0:
        total_steps = max_steps_arg
    else:
        total_steps = steps_per_epoch * num_train_epochs_arg

    # Floor at 40; otherwise 10% of total steps.
    warmup_steps = max(40, total_steps // 10) if total_steps > 40 else max(1, total_steps // 10)
    # Save checkpoint every epoch
    save_steps = steps_per_epoch
    # Trainer eval_loss every half epoch (the EvalCallback handles accuracy separately)
    eval_steps = max(50, steps_per_epoch // 2)

    if SMOKE_TEST:
        save_steps = 5
        eval_steps = 5

    print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
    print(f"Warmup: {warmup_steps}, save: every {save_steps} steps (≈1 epoch), eval_loss: every {eval_steps}\n")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,

            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_train_epochs_arg,
            max_steps=max_steps_arg,
            warmup_steps=warmup_steps,
            learning_rate=lr,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.01,

            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=NUM_EPOCHS + 1,
            seed=SEED,
            bf16=True,
            output_dir=str(run_dir / "checkpoints"),
            report_to=REPORT_TO,
            dataloader_num_workers=0,

            eval_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=max(1, batch_size // 4),

            # Early stopping needs these two set even when the callback is off,
            # so load_best_model_at_end works if toggled later. Safe no-ops otherwise.
            metric_for_best_model="eval_accuracy" if EARLY_STOPPING else None,
            greater_is_better=True if EARLY_STOPPING else None,
            load_best_model_at_end=EARLY_STOPPING,
        ),
    )

    # --- Callbacks ---
    loss_logger = LossLogger(run_dir)
    trainer.add_callback(loss_logger)

    eval_callback = EvalCallback(
        val_records=val_records,
        base_dir=str(split_dir),
        max_edge=CFG["image_max_edge"],
        run_dir=run_dir,
        steps_per_epoch=steps_per_epoch,
        baseline_accuracy=baseline_accuracy,
        use_wandb=use_wandb,
    )
    trainer.add_callback(eval_callback)

    # HF EarlyStoppingCallback watches `eval_accuracy` emitted by our EvalCallback
    # at each full-epoch eval. Patience=2 means: stop if 2 consecutive epochs
    # fail to improve over the best seen so far.
    if EARLY_STOPPING:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
        print("[early-stopping] enabled: patience=2 on eval_accuracy (greater is better)\n")

    # --- Train ---
    print("Starting training..." + (" (SMOKE TEST)" if SMOKE_TEST else ""))
    trainer_stats = trainer.train()

    # --- Plots ---
    plot_training_curves(loss_logger, eval_callback, run_dir, profile_name)

    # --- Save LoRA ---
    lora_dir = str(run_dir / "lora")
    print(f"\nSaving LoRA to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    if not SMOKE_TEST:
        merged_dir = str(run_dir / "merged")
        print(f"Saving merged bf16 to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # --- Final eval (finetuned only — base eval lives in eval_base.py) ---
    # We intentionally do NOT re-run a fake "base" eval here. For a real
    # base-vs-finetuned comparison use eval_base.py with and without
    # EVAL_ADAPTER pointing at the saved LoRA — that gives proper, greedy,
    # full-val numbers under the same code path as mid-training eval.
    print("\nFinal sample eval (3 random val samples, finetuned greedy)...")
    ft_eval = run_eval_samples(model, tokenizer, val_records, str(split_dir), CFG["image_max_edge"], n=3)
    for r in ft_eval:
        print(f"  {r['question_id']}: gold={r['gold']}, pred={r['predicted'][:40]!r}, "
              f"letter={r.get('letter')}, {'OK' if r['correct'] else 'WRONG'}")
    # Write finetuned-only markdown (base column intentionally omitted)
    with open(run_dir / "eval_samples.md", "w") as fh:
        fh.write("# Final Eval Samples (finetuned)\n\n")
        fh.write("For a fair base-vs-finetuned comparison, run `python eval_base.py` and\n")
        fh.write("then `EVAL_ADAPTER=<run_dir>/lora python eval_base.py` — identical code path.\n\n")
        for r in ft_eval:
            fh.write(f"## {r['question_id']} ({r['topic']})\n\n")
            fh.write(f"**Question:** {r['question']}\n\n")
            fh.write(f"**Gold:** {r['gold']}\n\n")
            fh.write(f"**Predicted:** {r['predicted'][:80]!r} → parsed `{r.get('letter')}` "
                     f"{'✓' if r['correct'] else '✗'}\n\n")

    # --- Summary ---
    t_total = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    peak_pct = peak_vram_mb / (CFG["vram_gb"] * 1024) * 100

    final_acc = eval_callback.full_history[-1]["overall"] if eval_callback.full_history else "N/A"
    final_acc_str = f"{final_acc:.1%}" if isinstance(final_acc, float) else final_acc

    summary = f"""
{'=' * 60}
TRAINING SUMMARY
{'=' * 60}
Profile:           {profile_name}
max_seq_length:    {max_seq_length} (p99={token_stats['p99']})
Batch:             {batch_size} x {grad_accum} = {effective_batch}
Epochs:            {NUM_EPOCHS}
Early stopping:    {EARLY_STOPPING}
Final train loss:  {trainer_stats.training_loss:.4f}
Final val accuracy:{final_acc_str}
Base accuracy:     {baseline_accuracy['overall']:.1%}
Wall clock:        {t_total/60:.1f} min
Peak VRAM:         {peak_vram_mb:.0f} MB ({peak_pct:.0f}% of {CFG['vram_gb']} GB)
Run dir:           {run_dir.name}
"""
    print(summary)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
