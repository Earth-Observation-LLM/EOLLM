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
    EARLY_STOPPING, NUM_EPOCHS_EXPLICIT, TEXT_ONLY,
    MODEL_FAMILY, MODEL_NAME, CHAT_TEMPLATE, SAVE_MERGED,
    COLLATOR_INSTRUCTION_PART, COLLATOR_RESPONSE_PART,
    detect_profile, find_dataset_dir, generate_run_dir, seed_everything,
)
from data import (
    load_jsonl,
    convert_record,
    measure_token_lengths,
    EollmDataset,
    select_probe_samples,
    cost_sorted_indices,
)
from evaluation import run_eval_samples, write_eval_md, compute_topic_accuracy, write_accuracy_md
from callbacks import LossLogger, EvalCallback, plot_training_curves


# ---------------------------------------------------------------------------
# Batch size probing — smart worst-case probe
# ---------------------------------------------------------------------------
#
# Design summary (see design notes in README / tuning_notes.md output):
#
#   1. Sample selection: `select_probe_samples` picks the K=8 worst real
#      records (by true token count through the real processor) + 1 synthetic
#      upper-bound sample (5 max-edge blank images). This is done in
#      data.select_probe_samples, not here — this function just measures VRAM
#      for the batches it's given.
#
#   2. Probe schedule: geometric climb from max(initial_batch_guess, 8) to
#      first-fail, then linear-fit prediction + binary search to land inside
#      the target utilization band.
#
#   3. Measurement: 1 warmup + 3 scored iterations per bs (max_memory_reserved,
#      not allocated — reserved is what actually occupies VRAM from the
#      driver's POV and what OOMs). A throwaway AdamW(lr=1e-9) is stepped so
#      the optimizer-state VRAM is materialized in the measurement.
#
#   4. Safety margin: tiered — 10% / 12% / 15% headroom by bs band, plus an
#      absolute PROBE_RESERVE_MB reserve (default 1024) for allocator overhead.
#      Larger bs has more activation-checkpoint variance and needs more slack.
#
#   5. OOM is recoverable: empty_cache + gc + synchronize + reset_peak; probe
#      loop continues with `hi = bs` instead of breaking out.


_PROBE_HARD_CAP_DEFAULT = 64           # absolute ceiling regardless of profile
_PROBE_REPEATS_DEFAULT = 3             # scored fwd+bwd+step iters per bs
_PROBE_RESERVE_MB_DEFAULT = 1024       # absolute VRAM reserve on top of % margin
_PROBE_TARGET_BAND_DEFAULT = (0.88, 0.95)  # accept bs if peak/budget in this band


def _vram_budget_mb_legacy(safety_frac: float) -> tuple[int, int, int]:
    """Legacy flat-margin budget used only when BATCH_SAFETY_FRAC env is set.

    Returns (budget_mb, free_mb, total_mb). Preserved for backward compat with
    SKIP_PROBE=1 callers that pre-existed the tiered-margin rewrite.
    """
    free_b, total_b = torch.cuda.mem_get_info()
    free_mb = free_b / 1024**2
    total_mb = total_b / 1024**2
    return int(free_mb * safety_frac), int(free_mb), int(total_mb)


# Kept as a thin backwards-compat wrapper — some branches (SKIP_PROBE) still
# call it. New code should use `_tiered_budget_mb`.
def _vram_budget_mb(safety_frac: float = 0.98) -> tuple[int, int, int]:
    return _vram_budget_mb_legacy(safety_frac)


def _tiered_margin(bs: int) -> float:
    """Adaptive VRAM safety margin that grows with bs.

    Larger batches have more activation-checkpoint recompute variance and more
    fragmentation risk, so we reserve proportionally more headroom.

    PROBE_FLAT_MARGIN env overrides the tiers with a single flat fraction. Set
    it to 0.0 (with a modest PROBE_RESERVE_MB, e.g. 2048) when you want the
    budget to be "free VRAM minus a fixed absolute reserve" and nothing more —
    i.e. push bs as high as fits with only ~reserve_mb headroom at worst case.
    This is the "speed" knob: bigger batches, thinner safety margin.
    """
    flat = os.environ.get("PROBE_FLAT_MARGIN")
    if flat is not None:
        return float(flat)
    if bs <= 8:
        return 0.10
    if bs <= 16:
        return 0.12
    return 0.15


def _tiered_budget_mb(total_mb: int, bs: int, reserve_mb: int) -> int:
    """Compute the per-bs VRAM budget, measured against TOTAL card VRAM.

    budget = (total_mb - reserve_mb) * (1 - tiered_margin(bs))

    CRITICAL: the budget is compared against `max_memory_reserved()`, which is
    the WHOLE process footprint — frozen base-model weights PLUS activations.
    So the budget MUST be measured against TOTAL VRAM, not free-after-load.
    Using free-after-load here double-counts the weights (they're in `peak`
    on the left but excluded from `free` on the right), which made the probe
    declare a 27B that peaks at 56 GB on a 97 GB card "147% over budget" and
    reject every bs down to 1. For a small model (4B, ~10 GB weights) the bug
    was invisible because peak never approached the (already large) budget; for
    a 53 GB-weight 27B it was fatal.

    The absolute reserve leaves headroom for allocator/fragmentation overhead
    on top of the measured peak.
    """
    usable = max(0, total_mb - reserve_mb)
    return int(usable * (1.0 - _tiered_margin(bs)))


def _run_one_probe_step(collator, model, optimizer, samples: list[dict], bs: int) -> None:
    """One fwd + bwd + optimizer.step at batch size `bs`, cycling samples if needed.

    Cycling the sample pool is fine because we measure VRAM, not gradients —
    even identical samples inflate activations the same way as distinct ones.
    """
    # Cycle the pool so bs > len(samples) still works.
    batch_records = [samples[i % len(samples)] for i in range(bs)]
    batch = collator(batch_records)
    batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Synchronize so async kernels flush their allocations into the peak stats
    # before we read them.
    torch.cuda.synchronize()


def _measure_peak_mb(
    collator,
    model,
    optimizer,
    samples: list[dict],
    bs: int,
    repeats: int,
) -> tuple[float, float]:
    """Warmup + `repeats` scored iters; return (peak_reserved_mb, peak_allocated_mb).

    Peak is the max across scored iters (not mean) — we want the worst case.
    Warmup iter is discarded to exclude one-time allocator overhead and
    cuDNN autotune for the new batch shape.
    """
    # Warmup (discarded)
    _run_one_probe_step(collator, model, optimizer, samples, bs)

    peaks_reserved: list[float] = []
    peaks_allocated: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _run_one_probe_step(collator, model, optimizer, samples, bs)
        peaks_reserved.append(torch.cuda.max_memory_reserved() / 1024**2)
        peaks_allocated.append(torch.cuda.max_memory_allocated() / 1024**2)
    return max(peaks_reserved), max(peaks_allocated)


def _reset_cuda_state() -> None:
    """Best-effort reset between probe tiers / after OOM."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


def _predict_max_bs(peaks: dict[int, float], budget_getter) -> int | None:
    """Linear fit peak(bs) ≈ a + b*bs on observed OK points; predict largest bs
    whose predicted peak is ≤ budget at that bs.

    Returns None if we have fewer than 2 points or slope is non-positive (noisy).
    `budget_getter(bs)` returns the bs-dependent budget — the tiered margin
    means we compare against a sliding ceiling, not a fixed one.
    """
    if len(peaks) < 2:
        return None
    xs = sorted(peaks.keys())
    ys = [peaks[x] for x in xs]
    # Ordinary least squares (two-point closed form works fine for 2-5 points)
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    b = num / den
    a = mean_y - b * mean_x
    if b <= 0:
        return None
    # Predicted max bs: largest integer bs where a + b*bs <= budget(bs).
    # Scan from last observed bs upward, short-circuit on miss.
    last = xs[-1]
    best = last
    for cand in range(last + 1, last * 4 + 1):
        predicted = a + b * cand
        if predicted <= budget_getter(cand):
            best = cand
        else:
            break
    return best


def probe_batch_size(
    model,
    tokenizer,
    probe_samples: list[dict],
    start_bs: int = 8,
    hard_cap: int = _PROBE_HARD_CAP_DEFAULT,
    repeats: int = _PROBE_REPEATS_DEFAULT,
    reserve_mb: int = _PROBE_RESERVE_MB_DEFAULT,
    target_band: tuple[float, float] = _PROBE_TARGET_BAND_DEFAULT,
    max_bisections: int = 4,
) -> dict:
    """Find the largest safe batch size for the current model + VRAM budget.

    Contract:
      - `probe_samples` MUST contain worst-case samples (see data.select_probe_samples).
        This function does not pick samples; it measures what it's given.
      - Starting bs defaults to 8. On cards where bs=1/2/4 would fit trivially,
        starting at 8 saves probe time. If the start bs OOMs, we back off to 1.
      - Returns a dict with full probe trace for tuning_notes.md logging.

    Strategy:
      Phase A (geometric climb): bs = start, 2*start, 4*start, ... until
                                 OOM / OVER_BUDGET / hard_cap.
      Phase B (refine):         linear-fit peak(bs), predict hi, binary-search
                                 into [last_ok_bs, min(predicted, 2*last_ok)]
                                 until we land in the target utilization band.
    """
    from unsloth.trainer import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator(model, tokenizer)

    # Build a throwaway AdamW over trainable params so optimizer.step materializes
    # the momentum/variance buffers during the probe — current peak would
    # otherwise undercount by GB once training actually starts.
    trainable = [p for p in model.parameters() if p.requires_grad]
    # lr=1e-9 means step() touches all the buffers but doesn't meaningfully move
    # weights. We discard this optimizer after the probe.
    optimizer = torch.optim.AdamW(trainable, lr=1e-9)

    _reset_cuda_state()
    free_b, total_b = torch.cuda.mem_get_info()
    free_mb = int(free_b / 1024**2)
    total_mb = int(total_b / 1024**2)

    # Budget is measured against TOTAL VRAM because peak_reserved (what we
    # compare against) already includes the resident weights. See
    # _tiered_budget_mb docstring — passing free_mb here double-counts weights.
    def budget_for(bs: int) -> int:
        return _tiered_budget_mb(total_mb, bs, reserve_mb)

    print(f"  VRAM: {free_mb} MB free / {total_mb} MB total "
          f"(model+overhead already using {total_mb - free_mb} MB)")
    print(f"  Budget vs TOTAL VRAM (peak_reserved includes resident weights)")
    print(f"  Reserve: {reserve_mb} MB absolute + tiered margin 10%/12%/15% by bs band")
    print(f"  Target band: [{target_band[0]*100:.0f}%, {target_band[1]*100:.0f}%] of budget")
    print(f"  Probe samples: {len(probe_samples)} (worst-case, see tuning_notes.md)")

    trace: dict[int, dict] = {}
    ok_peaks: dict[int, float] = {}  # bs -> peak_reserved_mb, for linear fit
    last_ok: int | None = None
    ceiling: int | None = None       # smallest OOMed / over-budget bs
    ceiling_reason: str | None = None

    def try_bs(bs: int) -> str:
        """Probe one bs; update trace / ok_peaks / ceiling. Returns status string."""
        nonlocal last_ok, ceiling, ceiling_reason
        if bs in trace:
            return trace[bs]["status"]
        _reset_cuda_state()
        budget = budget_for(bs)
        try:
            peak_r, peak_a = _measure_peak_mb(collator, model, optimizer, probe_samples, bs, repeats)
            util = peak_r / budget if budget > 0 else float("inf")
            status = "OK" if peak_r <= budget else "OVER_BUDGET"
            trace[bs] = {
                "status": status,
                "peak_reserved_mb": round(peak_r),
                "peak_allocated_mb": round(peak_a),
                "budget_mb": budget,
                "util": round(util, 3),
            }
            if status == "OK":
                last_ok = bs if (last_ok is None or bs > last_ok) else last_ok
                ok_peaks[bs] = peak_r
                print(f"  bs={bs:>3}: OK           peak_res {peak_r:6.0f} MB  "
                      f"(budget {budget} MB, util {util*100:.0f}%)")
            else:
                if ceiling is None or bs < ceiling:
                    ceiling = bs
                    ceiling_reason = "OVER_BUDGET"
                print(f"  bs={bs:>3}: OVER_BUDGET  peak_res {peak_r:6.0f} MB  "
                      f"(budget {budget} MB, util {util*100:.0f}%)")
            return status
        except torch.cuda.OutOfMemoryError:
            _reset_cuda_state()
            trace[bs] = {"status": "OOM", "budget_mb": budget}
            if ceiling is None or bs < ceiling:
                ceiling = bs
                ceiling_reason = "OOM"
            print(f"  bs={bs:>3}: OOM          (budget {budget} MB)")
            return "OOM"

    # ---- Phase A: geometric climb ----
    print("  Phase A — geometric climb:")
    bs = max(1, start_bs)
    while bs <= hard_cap:
        status = try_bs(bs)
        if status != "OK":
            break
        # Stop early if we're already inside the target band — no need to push further
        # and risk fragmenting the allocator.
        util = trace[bs]["util"]
        if target_band[0] <= util <= target_band[1]:
            print(f"  Phase A hit target band at bs={bs} (util={util*100:.0f}%) — skipping Phase B")
            break
        next_bs = bs * 2
        if next_bs > hard_cap:
            # Try the cap itself if we haven't
            if bs != hard_cap:
                try_bs(hard_cap)
            break
        bs = next_bs

    # If start bs itself failed, back off geometrically to find a working bs=1/2/4.
    if last_ok is None:
        print("  Start bs failed — backing off to find any working bs...")
        for fallback in (4, 2, 1):
            if fallback >= start_bs:
                continue
            if try_bs(fallback) == "OK":
                break

    if last_ok is None:
        raise RuntimeError(
            "Probe failed at every tested bs including 1. Cannot recover — "
            "training would crash on first step. Free more VRAM, reduce "
            "image_max_edge, or disable vision layer training."
        )

    # ---- Phase B: linear-fit prediction + binary refinement ----
    # Skip if we already hit the target band, or if Phase A never failed (last_ok == hard_cap).
    in_band = target_band[0] <= trace[last_ok]["util"] <= target_band[1]
    if not in_band and ceiling is not None:
        predicted = _predict_max_bs(ok_peaks, budget_for)
        # Binary-search upper bound: don't go beyond ceiling-1 or predicted value.
        hi_cap = ceiling - 1 if ceiling else hard_cap
        if predicted is not None:
            hi_cap = min(hi_cap, predicted)
        hi_cap = min(hi_cap, hard_cap)

        if hi_cap > last_ok:
            print(f"  Phase B — binary refine in [{last_ok+1}, {hi_cap}] "
                  f"(linear fit predicts max≈{predicted})")
            lo, hi = last_ok, hi_cap + 1  # hi exclusive
            for _ in range(max_bisections):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                if mid == lo or mid == hi:
                    break
                status = try_bs(mid)
                if status == "OK":
                    lo = mid
                    util = trace[mid]["util"]
                    if target_band[0] <= util <= target_band[1]:
                        print(f"  Phase B hit target band at bs={mid} (util={util*100:.0f}%)")
                        break
                else:
                    hi = mid
        else:
            print(f"  Phase B skipped — last_ok={last_ok} already at ceiling - 1")
    elif in_band:
        print(f"  Phase B skipped — already in target band at bs={last_ok}")

    _reset_cuda_state()

    chosen = last_ok
    chosen_info = trace[chosen]
    peak_util = chosen_info.get("util")

    # Emit a slope summary for tuning_notes.md and log readability
    slope_mb_per_bs = None
    if len(ok_peaks) >= 2:
        xs = sorted(ok_peaks.keys())
        peaks_list = [ok_peaks[x] for x in xs]
        slope_mb_per_bs = (peaks_list[-1] - peaks_list[0]) / (xs[-1] - xs[0])

    return {
        "probed": trace,
        "recommended_batch_size": chosen,
        "peak_at_recommended_mb": chosen_info.get("peak_reserved_mb"),
        "peak_util": peak_util,
        "budget_mb": chosen_info.get("budget_mb"),
        "free_mb": free_mb,
        "total_mb": total_mb,
        "reserve_mb": reserve_mb,
        "margin_applied": _tiered_margin(chosen),
        "target_band": target_band,
        "hard_cap": hard_cap,
        "ceiling_bs": ceiling,
        "ceiling_reason": ceiling_reason,
        "scaling_slope_mb_per_bs": slope_mb_per_bs,
        # safety_frac is not a single number in the tiered scheme; keep the key
        # for backward compat with the tuning_notes writer.
        "safety_frac": 1.0 - _tiered_margin(chosen),
    }


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
    print(f"EOLLM Vision SFT — {MODEL_NAME}")
    print("=" * 60)
    print(f"Model:            {MODEL_NAME}  (family={MODEL_FAMILY})")
    print(f"Base model:       {CFG_BASE_MODEL}")
    print(f"Profile:          {profile_name}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Image max edge:   {CFG['image_max_edge']} px")
    print(f"LoRA r/alpha:     {CFG['lora_r']}/{CFG['lora_alpha']}")
    print(f"LR:               {lr}")
    # Text-only baseline: no image inputs ever reach the model, so training
    # vision layers is wasted parameters (and a tiny risk if the LoRA on the
    # visual encoder shifts the unused subgraph). Force off in TEXT_ONLY.
    finetune_vision_layers = False if TEXT_ONLY else CFG["finetune_vision_layers"]
    print(f"Vision layers:    {finetune_vision_layers}{' (forced off — TEXT_ONLY)' if TEXT_ONLY else ''}")
    print(f"Text-only mode:   {TEXT_ONLY}")
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
                    "model": MODEL_NAME, "model_family": MODEL_FAMILY,
                    "method": "bf16 LoRA",
                    "profile": profile_name, "image_max_edge": CFG["image_max_edge"],
                    "lora_r": CFG["lora_r"], "lora_alpha": CFG["lora_alpha"],
                    "lr": lr, "finetune_vision_layers": finetune_vision_layers,
                    "split": SPLIT, "seed": SEED, "num_epochs": NUM_EPOCHS,
                    "early_stopping": EARLY_STOPPING, "text_only": TEXT_ONLY,
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

    # EXCLUDE_TOPICS=a,b drops topics from TRAIN only. Eval keeps all topics so
    # we can still see per-topic numbers on the excluded ones (expect them to
    # stay pinned to their base-model level).
    exclude = {t.strip() for t in os.environ.get("EXCLUDE_TOPICS", "").split(",") if t.strip()}
    if exclude:
        before = len(train_records)
        train_records = [r for r in train_records if r.get("topic") not in exclude]
        print(f"  EXCLUDE_TOPICS={sorted(exclude)}: train {before} → {len(train_records)}")

    print(f"  Train: {len(train_records)}, Val: {len(val_records)}\n")

    # --- Pre-flight VRAM guard ---
    # If free VRAM is too low, accelerate silently CPU-offloads part of the
    # vision tower — that offload path uses AlignDevicesHook which is wrapped
    # with @torch.compiler.disable in accelerate 1.13+, and Unsloth's compiled
    # vision forward crashes trying to trace through it. Better to abort with
    # a clear message than to silently slide into a broken run.
    free_mb_pre, total_mb_pre = torch.cuda.mem_get_info()
    free_gb_pre = free_mb_pre / 1024**3
    total_gb_pre = total_mb_pre / 1024**3
    MIN_FREE_GB_FOR_LOAD = 20.0
    print(f"VRAM pre-flight:  {free_gb_pre:.1f} GB free / {total_gb_pre:.1f} GB total")
    if free_gb_pre < MIN_FREE_GB_FOR_LOAD:
        raise RuntimeError(
            f"Only {free_gb_pre:.1f} GB free (need ≥{MIN_FREE_GB_FOR_LOAD} GB for model load). "
            f"Another process is holding {total_gb_pre - free_gb_pre:.1f} GB. "
            f"Waiting is much cheaper than silent CPU offload — the offload path crashes "
            f"Unsloth's compiled vision forward. Rerun when the GPU is freer."
        )

    # --- Load model ---
    print(f"Loading {MODEL_NAME} (bf16 LoRA)...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CFG_BASE_MODEL,
        load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8192,
    )

    # Gemma-4 needs its chat template applied to the processor explicitly (the
    # unsloth notebook does get_chat_template(proc, "gemma-4")). Qwen3.5 ships a
    # working template already — CHAT_TEMPLATE is None for it and we skip this.
    if CHAT_TEMPLATE is not None:
        from unsloth import get_chat_template
        tokenizer = get_chat_template(tokenizer, CHAT_TEMPLATE)
        print(f"Applied chat template: {CHAT_TEMPLATE}")

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
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=0, bias="none",
        target_modules="all-linear", random_state=SEED,
    )
    print("LoRA attached.\n")

    # --- Probe samples — worst-case batches for accurate VRAM estimation ---
    # The selector does 2-stage filtering (cheap image-count+text score across
    # all records → top-N → tokenize top-K by real token length) and appends a
    # synthetic upper-bound sample (5 max-edge blank images). This replaces the
    # old "first-32 heavy-mode records" approach, which under-ranked records
    # whose real token count was higher than their mode implied.
    print("Selecting worst-case probe samples...")
    probe_samples, selector_info = select_probe_samples(
        train_records,
        base_dir=str(split_dir),
        max_edge=CFG["image_max_edge"],
        processor=tokenizer,
        k_real=int(os.environ.get("PROBE_K_REAL", "8")),
        stage_a_topn=int(os.environ.get("PROBE_STAGE_A_TOPN", "200")),
        include_synthetic=os.environ.get("PROBE_INCLUDE_SYNTHETIC", "1") == "1",
    )
    n_imgs_per = [sum(1 for p in s["messages"][1]["content"] if p.get("type") == "image")
                  for s in probe_samples]
    print(f"  Selected {selector_info['k_real']} real + "
          f"{'1 synthetic' if selector_info['synthetic_included'] else '0 synthetic'} "
          f"= {len(probe_samples)} probe samples")
    print(f"  Real token lengths (top-K): {selector_info['token_lengths']}")
    if selector_info["synthetic_included"]:
        print(f"  Synthetic token length:  {selector_info['synthetic_token_length']}")
    print(f"  Images per sample: min={min(n_imgs_per)}, max={max(n_imgs_per)}")

    # --- Probe batch size (or skip to claim GPU fast) ---
    FastVisionModel.for_training(model)
    skip_probe = os.environ.get("SKIP_PROBE", "0") == "1"
    forced_bs = int(os.environ.get("BATCH_SIZE", "0"))

    if skip_probe:
        # Fast claim: bypass probe entirely. User has asserted via BATCH_SIZE
        # that they've pre-measured this bs fits. The only contention check
        # that still matters is "is another user holding significant VRAM
        # beyond what the model expected" — but the model+LoRA alone takes
        # 30% of a 32GB card, so a flat 90%-free rule is useless here. We
        # only abort if free VRAM drops below ~40% of total, indicating a
        # second user or a leak (not normal model load).
        free_mb_now, total_mb_now = torch.cuda.mem_get_info()
        free_pct = free_mb_now / total_mb_now
        if free_pct < 0.40:
            raise RuntimeError(
                f"SKIP_PROBE=1 aborted — only {free_pct:.0%} of VRAM free post-load "
                f"({free_mb_now/1024**2:.0f} MB free of {total_mb_now/1024**2:.0f} MB). "
                f"Either another process is holding VRAM or there's a leak. "
                f"Free the GPU or run the probe."
            )
        if forced_bs <= 0:
            raise RuntimeError("SKIP_PROBE=1 requires BATCH_SIZE=<n> to be set.")
        batch_size = forced_bs
        safety_frac = float(os.environ.get("BATCH_SAFETY_FRAC", "0.98"))
        budget_mb, _free_mb, total_mb = _vram_budget_mb(safety_frac)
        probe_result = {
            "probed": {forced_bs: {"status": "SKIPPED (forced)", "peak_reserved_mb": "—"}},
            "recommended_batch_size": forced_bs,
            "peak_at_recommended_mb": None,
            "peak_util": None,
            "budget_mb": budget_mb,
            "free_mb": int(free_mb_now / 1024**2),
            "total_mb": total_mb,
            "reserve_mb": 0,
            "margin_applied": 1.0 - safety_frac,
            "target_band": (0.0, 1.0),
            "hard_cap": forced_bs,
            "ceiling_bs": None,
            "ceiling_reason": None,
            "scaling_slope_mb_per_bs": None,
            "safety_frac": safety_frac,
        }
        print(f"[probe] SKIPPED — forced batch_size={forced_bs} (gate: {free_pct:.0%} free VRAM)")
    else:
        # New tiered-margin probe. Env overrides:
        #   PROBE_START_BS     — default = max(CFG.initial_batch_guess, 8)
        #   PROBE_HARD_CAP     — default 64 (absolute upper bound)
        #   PROBE_REPEATS      — default 3 scored iters per bs
        #   PROBE_RESERVE_MB   — default 1024 MB absolute reserve
        #   PROBE_TARGET_LOW   — default 0.88 (accept if util within [low, high])
        #   PROBE_TARGET_HIGH  — default 0.95
        start_bs = int(os.environ.get("PROBE_START_BS", max(CFG["initial_batch_guess"], 8)))
        hard_cap = int(os.environ.get("PROBE_HARD_CAP", _PROBE_HARD_CAP_DEFAULT))
        repeats = int(os.environ.get("PROBE_REPEATS", _PROBE_REPEATS_DEFAULT))
        reserve_mb = int(os.environ.get("PROBE_RESERVE_MB", _PROBE_RESERVE_MB_DEFAULT))
        target_low = float(os.environ.get("PROBE_TARGET_LOW", _PROBE_TARGET_BAND_DEFAULT[0]))
        target_high = float(os.environ.get("PROBE_TARGET_HIGH", _PROBE_TARGET_BAND_DEFAULT[1]))

        print(f"Probing batch size "
              f"(start_bs={start_bs}, hard_cap={hard_cap}, repeats={repeats}, "
              f"reserve={reserve_mb} MB, target=[{target_low:.2f},{target_high:.2f}])...")
        probe_result = probe_batch_size(
            model, tokenizer, probe_samples,
            start_bs=start_bs, hard_cap=hard_cap, repeats=repeats,
            reserve_mb=reserve_mb, target_band=(target_low, target_high),
        )
        batch_size = probe_result["recommended_batch_size"]

    # GRAD_ACCUM env var lets the user pin gradient_accumulation_steps
    # explicitly (useful when BATCH_SIZE is forced). Otherwise we auto-compute
    # so effective_batch ≈ 24-32 depending on dataset size.
    _env_grad_accum = int(os.environ.get("GRAD_ACCUM", "0"))
    if _env_grad_accum > 0:
        grad_accum = _env_grad_accum
    else:
        target_effective = max(8, min(32, len(train_records) // 1000))
        grad_accum = max(1, target_effective // batch_size)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = math.ceil(len(train_records) / effective_batch)

    print(f"\n  bs={batch_size} x grad_accum={grad_accum} = {effective_batch} effective")
    print(f"  steps/epoch: {steps_per_epoch}\n")

    with open(run_dir / "tuning_notes.md", "w") as f:
        f.write("# Tuning Notes\n\n")
        f.write(f"max_seq_length: {max_seq_length}\n\n")
        f.write("## Probe Sample Selection\n\n")
        f.write(f"- records scanned: {selector_info['n_records']}\n")
        f.write(f"- stage-A top-N: {selector_info['stage_a_topn']}\n")
        f.write(f"- real probe samples (K): {selector_info['k_real']}\n")
        f.write(f"- real sample token lengths: {selector_info['token_lengths']}\n")
        f.write(f"- max real token length: {selector_info['max_token_length']}\n")
        f.write(f"- synthetic sample included: {selector_info['synthetic_included']}\n")
        if selector_info["synthetic_included"]:
            f.write(f"- synthetic token length: {selector_info['synthetic_token_length']}\n")
        f.write("\n## Probe Trace\n\n")
        f.write("| BS | Status | Peak Reserved (MB) | Peak Allocated (MB) | Budget (MB) | Util |\n")
        f.write("|----|--------|--------------------|---------------------|-------------|------|\n")
        for bs, info in sorted(probe_result["probed"].items()):
            util = info.get("util")
            util_str = f"{util*100:.0f}%" if isinstance(util, (int, float)) else "—"
            f.write(
                f"| {bs} | {info['status']} | "
                f"{info.get('peak_reserved_mb', info.get('peak_vram_mb', '—'))} | "
                f"{info.get('peak_allocated_mb', '—')} | "
                f"{info.get('budget_mb', '—')} | {util_str} |\n"
            )
        f.write("\n## Summary\n\n")
        f.write(f"- chosen bs: **{batch_size}**\n")
        f.write(f"- peak at chosen bs: {probe_result.get('peak_at_recommended_mb', '—')} MB reserved\n")
        peak_util = probe_result.get("peak_util")
        if isinstance(peak_util, (int, float)):
            f.write(f"- utilization: {peak_util*100:.1f}% of budget\n")
        f.write(f"- budget at chosen bs: {probe_result.get('budget_mb', '—')} MB\n")
        f.write(f"- free VRAM at probe: {probe_result.get('free_mb', '—')} MB / "
                f"{probe_result.get('total_mb', '—')} MB total\n")
        f.write(f"- absolute reserve: {probe_result.get('reserve_mb', '—')} MB\n")
        margin = probe_result.get("margin_applied")
        if isinstance(margin, (int, float)):
            f.write(f"- margin applied at chosen bs: {margin*100:.0f}% (tiered)\n")
        if probe_result.get("ceiling_bs") is not None:
            f.write(
                f"- ceiling detected at bs={probe_result['ceiling_bs']} "
                f"({probe_result.get('ceiling_reason', 'unknown')})\n"
            )
        if probe_result.get("scaling_slope_mb_per_bs") is not None:
            f.write(
                f"- VRAM slope: {probe_result['scaling_slope_mb_per_bs']:.1f} MB per bs unit\n"
            )
        f.write(f"- grad_accum: {grad_accum}, effective batch: {effective_batch}\n")
        f.write(f"- steps/epoch: {steps_per_epoch}\n")

    # --- Label masking check ---
    print("Checking label masking...")
    from unsloth.trainer import UnslothVisionDataCollator

    # Instruction/response markers are FAMILY-SPECIFIC (Qwen <|im_start|> vs
    # Gemma <|turn>). Wrong strings → train_on_responses_only masks the wrong
    # span → silently broken training. check_label_masking() below is the guard:
    # it sys.exit(1)s if masking produces all-masked or nothing-masked labels.
    print(f"  collator markers (family={MODEL_FAMILY}): "
          f"instruction={COLLATOR_INSTRUCTION_PART!r} response={COLLATOR_RESPONSE_PART!r}")
    collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part=COLLATOR_INSTRUCTION_PART,
        response_part=COLLATOR_RESPONSE_PART,
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

    # Baseline eval ran `FastVisionModel.for_inference(model)` and 100 greedy
    # generations, which allocates substantial KV-cache VRAM that the allocator
    # won't free on its own. Switch the model back to training mode and
    # aggressively reset the CUDA cache — otherwise the first training step
    # OOMs because 10+ GB of KV cache is still held alongside training
    # activations. Observed empirically: without this, a probe-chosen bs that
    # measured 18 GB peak ended up trying to run training on a card with only
    # ~1 GB free, because baseline-eval residue occupied the rest.
    from unsloth import FastVisionModel as _FVM
    _FVM.for_training(model)
    _reset_cuda_state()
    post_eval_free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"  Post-eval VRAM: {post_eval_free_mb:.0f} MB free (reset cache, model back in training mode)")

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
    _env_warmup = int(os.environ.get("WARMUP_STEPS", "0"))
    if _env_warmup > 0:
        warmup_steps = _env_warmup
    else:
        warmup_steps = max(40, total_steps // 10) if total_steps > 40 else max(1, total_steps // 10)
    # Eval exactly once per epoch. save_steps MUST equal eval_steps when
    # load_best_model_at_end=True — otherwise the "best" model can fall on a
    # non-save step and the final load fails.
    eval_steps = steps_per_epoch
    save_steps = eval_steps

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
            lr_scheduler_type=os.environ.get("LR_SCHEDULER", "cosine"),
            # adamw_8bit (default) halves optimizer-state VRAM vs adamw_torch —
            # essential headroom for bf16 27B/31B on a 96 GB card. The unsloth
            # vision notebooks use adamw_8bit. Override with OPTIM=adamw_torch
            # if you specifically want full-precision optimizer states.
            optim=os.environ.get("OPTIM", "adamw_8bit"),
            weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")),
            max_grad_norm=1.0,

            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=2 * NUM_EPOCHS + 1,
            seed=SEED,
            bf16=True,
            output_dir=str(run_dir / "checkpoints"),
            report_to=REPORT_TO,
            dataloader_num_workers=12,
            dataloader_persistent_workers=True,

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
    # at every eval_steps boundary (now one eval per full epoch). patience=2 →
    # stop if 2 consecutive epochs fail to improve.
    if EARLY_STOPPING:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
        print("[early-stopping] enabled: patience=2 epochs on eval_accuracy\n")

    # --- Pre-train VRAM sanity check (optional) ---
    # The probe already did fwd+bwd+optimizer.step across 1 warmup + 3 scored
    # iterations on worst-case samples, so the chosen bs is well-measured.
    # This sanity check is a lightweight smoke test: 1 fwd+bwd at the chosen
    # bs on the real training collator (with label-masking) to catch any
    # mismatch between probe collator and training collator. Disabled by
    # default now — the probe is authoritative. Enable with PROBE_STRESS_TEST=1
    # if you want the extra belt-and-braces check.
    if os.environ.get("PROBE_STRESS_TEST", "0") == "1" and not skip_probe:
        print(f"[stress] Sanity check: 1 fwd+bwd at chosen bs={batch_size} with training collator...")
        try:
            _reset_cuda_state()
            torch.cuda.reset_peak_memory_stats()
            free_now = torch.cuda.mem_get_info()[0] / 1024**2
            print(f"[stress] Free VRAM before stress: {free_now:.0f} MB")
            stress_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-9)
            # Use the REAL training collator to match label masking behavior.
            _run_one_probe_step(collator, model, stress_opt, probe_samples, batch_size)
            peak_mb = torch.cuda.max_memory_reserved() / 1024**2
            print(f"[stress] OK — peak reserved {peak_mb:.0f} MB at bs={batch_size}")
            del stress_opt
            _reset_cuda_state()
        except torch.cuda.OutOfMemoryError:
            _reset_cuda_state()
            raise RuntimeError(
                f"Stress check OOMed at chosen bs={batch_size}. Probe was too "
                f"aggressive or VRAM was consumed between probe and training "
                f"(e.g. baseline eval didn't free its cache). "
                f"Lower PROBE_TARGET_HIGH or raise PROBE_RESERVE_MB and restart."
            )
        except Exception as e:
            # Non-OOM failures: warn but continue. Probe already passed.
            print(f"[stress] WARNING — stress check threw {type(e).__name__}: {e}")
            _reset_cuda_state()

    # --- Train ---
    # Auto-resume if a checkpoint exists in run_dir/checkpoints. Only applies
    # when OUTPUT_DIR was explicitly set (so the user is deliberately reusing
    # a run dir). Fresh timestamped runs have no checkpoints yet → fresh train.
    checkpoint_dir = run_dir / "checkpoints"
    has_ckpt = checkpoint_dir.exists() and any(checkpoint_dir.glob("checkpoint-*"))
    resume = has_ckpt
    if resume:
        print(f"[resume] Found existing checkpoint(s) in {checkpoint_dir} — resuming.")
    print("Starting training..." + (" (SMOKE TEST)" if SMOKE_TEST else ""))
    trainer_stats = trainer.train(resume_from_checkpoint=resume)

    # --- Plots ---
    plot_training_curves(loss_logger, eval_callback, run_dir, profile_name)

    # --- Save LoRA ---
    lora_dir = str(run_dir / "lora")
    print(f"\nSaving LoRA to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Merged 16-bit save is OFF by default (SAVE_MERGED=0). For 27B/31B the
    # merged model is ~52-62 GB EACH — 12 runs would blow the disk. The eval
    # path (eval_base.py with EVAL_ADAPTER) loads base+adapter directly, so the
    # merged model is never needed for evaluation. The LoRA adapter (~200 MB) is
    # always saved above. Set SAVE_MERGED=1 to force a merged export.
    if not SMOKE_TEST and SAVE_MERGED:
        merged_dir = str(run_dir / "merged")
        print(f"Saving merged bf16 to {merged_dir}...")
        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        except Exception as e:
            # Merge can fail (OOM during the merge spike, disk full, etc.) but
            # the LoRA above is already on disk — don't let this kill the run
            # before summary.txt is written.
            print(f"[warn] save_pretrained_merged failed: {e}")
            print("[warn] LoRA adapter is saved — you can merge later with eval_base.py")
    elif not SMOKE_TEST:
        print("Skipping merged save (SAVE_MERGED=0). LoRA adapter saved; "
              "eval via eval_base.py with EVAL_ADAPTER loads base+adapter directly.")

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
