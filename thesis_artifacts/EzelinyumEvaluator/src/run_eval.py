"""Main evaluation entrypoint.

For one model, evaluate every benchmark record. Write:
- predictions.jsonl: one row per (question_id, model) with full provenance.
- report.json: per-topic / per-city / per-difficulty / per-land_use metrics.
- meta.json: seed, git SHA, model revision, wall time, peak VRAM, etc.

Addresses every audit finding from the EarthMLLMEval review:
- Full seeding (torch, numpy, random, CUDA, cudnn.deterministic)
- LoRA loading via Unsloth + PEFT
- Per-record options for F1/AUC (in metrics.py)
- Sum-not-max logit-prob extraction
- Hedging detection
- Missing-image accounting
- Git SHA / model revision logging
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure src is on the path.
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from registry import get as get_spec  # noqa
from dataio import load_benchmark, select_images_for_record, build_prompt, valid_letters  # noqa
from parsing import parse_answer  # noqa
from metrics import per_topic_baselines, per_topic_metrics, wilson_ci  # noqa


def seed_everything(seed: int = 3407) -> None:
    """Set every seed we know about and enforce deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # `use_deterministic_algorithms` can break attention kernels; opt-in via env.
    if os.environ.get("STRICT_DETERMINISM") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)


def _safe_generate_with_oom_retry(generate_batch, model, processor, gen_inputs, max_new_tokens):
    """Try batched generation. On CUDA OOM, halve the batch and recurse.
    Falls through to per-record on non-OOM failures.
    """
    if not gen_inputs:
        return []
    try:
        return generate_batch(model, processor, gen_inputs, max_new_tokens=max_new_tokens)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        if len(gen_inputs) == 1:
            print(f"[warn] OOM on batch=1: {e}; returning empty for this record", flush=True)
            return [("<oom>", {})]
        half = max(1, len(gen_inputs) // 2)
        print(f"[warn] OOM at batch={len(gen_inputs)}, halving to {half}+{len(gen_inputs)-half}", flush=True)
        left = _safe_generate_with_oom_retry(generate_batch, model, processor, gen_inputs[:half], max_new_tokens)
        right = _safe_generate_with_oom_retry(generate_batch, model, processor, gen_inputs[half:], max_new_tokens)
        return left + right
    except Exception as e:
        # Non-OOM error: try per-record so one bad sample doesn't lose the batch.
        print(f"[warn] batch generate failed (non-OOM) at size={len(gen_inputs)}: {e}; falling back per-record", flush=True)
        out = []
        for gi in gen_inputs:
            try:
                r = generate_batch(model, processor, [gi], max_new_tokens=max_new_tokens)
                out.append(r[0] if r else ("", {}))
            except Exception as e2:
                out.append((f"<error: {e2}>", {}))
        return out


def git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model key from registry")
    ap.add_argument("--outdir", default="/home/ain480/EzelinyumEvaluator/outputs")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only eval this many records (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--image-mode", default="sat+stv4", choices=["sat+stv4", "sat_only", "stv_only"])
    ap.add_argument("--batch-size", type=int, default=128, help="Inference batch size; halves on OOM and retries")
    args = ap.parse_args()

    seed_everything(args.seed)

    spec = get_spec(args.model)
    outdir = Path(args.outdir) / spec.key
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path = outdir / "predictions.jsonl"
    report_path = outdir / "report.json"
    meta_path = outdir / "meta.json"

    # Load benchmark.
    records = load_benchmark()
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[eval] {spec.display}: {len(records)} records to evaluate", flush=True)

    # Load model.
    t_load_start = time.time()
    if spec.backend == "unsloth":
        from backend_unsloth import load_unsloth_model, generate_batch_unsloth as generate_batch
        model, processor = load_unsloth_model(spec.hf_id, lora_path=spec.lora_path)
    elif spec.backend == "hf":
        from backend_hf import load_hf_model, generate_batch_hf as generate_batch
        model, processor = load_hf_model(spec.hf_id)
    else:
        raise ValueError(f"Unknown backend {spec.backend}")
    t_load = time.time() - t_load_start
    print(f"[eval] model loaded in {t_load:.1f}s, batch_size={args.batch_size}", flush=True)

    # Eval loop — batched.
    t_eval_start = time.time()
    rows = []
    n_missing_total = 0
    n_load_fail = 0

    pred_path_tmp = pred_path.with_suffix(".jsonl.tmp")
    pending: list[dict] = []  # batch buffer of {record, images, roles, n_missing, prompt, valid_letters}

    def flush_pending():
        nonlocal pending
        if not pending:
            return
        gen_inputs = [
            dict(images=p["images"], prompt=p["prompt"], valid_letters=p["valid_letters"])
            for p in pending
        ]
        outputs = _safe_generate_with_oom_retry(
            generate_batch, model, processor, gen_inputs, args.max_new_tokens
        )

        for p, (raw, prob_dict) in zip(pending, outputs):
            r = p["record"]
            v_letters = p["valid_letters"]
            parsed = parse_answer(raw, set(v_letters))
            correct = (parsed.letter == r["answer"]) if parsed.letter else False
            row = dict(
                question_id=r["question_id"],
                sample_id=r["sample_id"],
                topic=r["topic"],
                difficulty=r.get("difficulty"),
                city=r.get("city"),
                country=r.get("country"),
                land_use=r.get("land_use"),
                city_type=r.get("city_type"),
                benchmark_city_type=r.get("benchmark_city_type"),
                generation_method=r.get("generation_method"),
                options=r["options"],
                gold=r["answer"],
                pred=parsed.letter,
                correct=correct,
                hedged=parsed.hedged,
                refused=parsed.refused,
                parse_path=parsed.parse_path,
                raw_response=raw,
                prob_dict=prob_dict,
                image_roles_used=p["roles"],
                n_missing_images=p["n_missing"],
                mismatch_strategy=r.get("mismatch_strategy"),
                mismatch_is_match=r.get("mismatch_is_match"),
            )
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
        fout.flush()
        pending = []

    with open(pred_path_tmp, "w") as fout:
        pbar = tqdm(records, desc=spec.key)
        for r in pbar:
            try:
                imgs, roles, n_missing = select_images_for_record(r, mode=args.image_mode)
                n_missing_total += n_missing
            except Exception as e:
                imgs, roles, n_missing = [], [], 0
                n_load_fail += 1
                print(f"[warn] image load failed for {r['question_id']}: {e}", flush=True)

            prompt = build_prompt(r)
            v_letters = sorted(valid_letters(r))
            pending.append(dict(
                record=r, images=imgs, roles=roles, n_missing=n_missing,
                prompt=prompt, valid_letters=v_letters,
            ))
            if len(pending) >= args.batch_size:
                flush_pending()
        flush_pending()
    # Atomic rename.
    pred_path_tmp.rename(pred_path)
    t_eval = time.time() - t_eval_start
    print(f"[eval] generation done in {t_eval/60:.1f} min", flush=True)

    # Per-record options carried in row["options"]; metrics consumer reads from there.
    # Build the report.
    report = build_report(records, rows)
    report["model"] = dict(
        key=spec.key,
        display=spec.display,
        backend=spec.backend,
        hf_id=spec.hf_id,
        lora_path=spec.lora_path,
        family=spec.family,
        notes=spec.notes,
    )
    report["counts"] = dict(
        total=len(records),
        evaluated=len(rows),
        n_missing_images=n_missing_total,
        n_load_fail=n_load_fail,
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] wrote {report_path}", flush=True)

    # Meta.
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
    meta = dict(
        model_key=spec.key,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        image_mode=args.image_mode,
        wall_load_sec=round(t_load, 1),
        wall_eval_sec=round(t_eval, 1),
        peak_vram_mb=peak_vram_mb,
        git_sha=git_sha(SRC),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        deterministic_algorithms=os.environ.get("STRICT_DETERMINISM") == "1",
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[eval] wrote {meta_path}", flush=True)


def build_report(records: list[dict], rows: list[dict]) -> dict:
    """Compute every metric we want in the headline report."""
    baselines = per_topic_baselines(records)

    # Overall.
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)
    overall = dict(
        n=n, correct=correct, accuracy=acc, ci95_low=lo, ci95_high=hi,
        n_hedged=sum(1 for r in rows if r.get("hedged")),
        n_refused=sum(1 for r in rows if r.get("refused")),
        n_unparseable=sum(1 for r in rows if r["pred"] is None and not r.get("refused")),
        n_missing_images=sum(r.get("n_missing_images", 0) for r in rows),
    )

    # Per-topic.
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    per_topic = {t: per_topic_metrics(rs) for t, rs in by_topic.items()}
    for t, m in per_topic.items():
        if t in baselines:
            m["random_baseline"] = baselines[t]["random_baseline"]
            m["majority_baseline"] = baselines[t]["majority_baseline"]
            m["avg_options"] = baselines[t]["avg_options"]
            m["delta_vs_random"] = m["accuracy"] - baselines[t]["random_baseline"]
            m["delta_vs_majority"] = m["accuracy"] - baselines[t]["majority_baseline"]

    # Per-city.
    by_city: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_city[r.get("city") or "_unknown"].append(r)
    per_city = {}
    for c, rs in by_city.items():
        nn = len(rs)
        cc = sum(1 for r in rs if r["correct"])
        lo_c, hi_c = wilson_ci(cc, nn)
        per_city[c] = dict(n=nn, correct=cc, accuracy=cc / nn, ci95_low=lo_c, ci95_high=hi_c)

    # Per-difficulty.
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r.get("difficulty") or "_unknown"].append(r)
    per_diff = {}
    for d, rs in by_diff.items():
        nn = len(rs)
        cc = sum(1 for r in rs if r["correct"])
        lo_d, hi_d = wilson_ci(cc, nn)
        per_diff[d] = dict(n=nn, correct=cc, accuracy=cc / nn, ci95_low=lo_d, ci95_high=hi_d)

    # Per-land_use.
    by_lu: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lu[r.get("land_use") or "_unknown"].append(r)
    per_lu = {}
    for lu, rs in by_lu.items():
        nn = len(rs)
        cc = sum(1 for r in rs if r["correct"])
        lo_l, hi_l = wilson_ci(cc, nn)
        per_lu[lu] = dict(n=nn, correct=cc, accuracy=cc / nn, ci95_low=lo_l, ci95_high=hi_l)

    # Per-city_type (seen vs unseen).
    by_ct: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ct[r.get("city_type") or "_unknown"].append(r)
    per_ct = {}
    for ct, rs in by_ct.items():
        nn = len(rs)
        cc = sum(1 for r in rs if r["correct"])
        lo_t, hi_t = wilson_ci(cc, nn)
        per_ct[ct] = dict(n=nn, correct=cc, accuracy=cc / nn, ci95_low=lo_t, ci95_high=hi_t)

    return dict(
        overall=overall,
        per_topic=per_topic,
        per_city=per_city,
        per_difficulty=per_diff,
        per_land_use=per_lu,
        per_city_type=per_ct,
        topic_baselines=baselines,
    )


if __name__ == "__main__":
    main()
