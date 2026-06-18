#!/usr/bin/env python3
"""Batched, multi-mode benchmark eval — vLLM backend, THINKING OFF.

Runs the EOLLM benchmark (benchmark_with_answers.jsonl, 5240 recs) through a
vision-language model under four image-ablation MODES, and writes a per-mode
predictions JSONL + metrics report (reusing metrics.build_full_report).

MODES (which images the model sees):
  full      everything the topic provides (sat + stv)         -> ALL topics
  blind     NO images at all (text-only control)              -> ALL topics
  sat_only  satellite image(s) only                           -> mixed-perspective topics
  sv_only   street-view image(s) only                         -> mixed-perspective topics

"Mixed-perspective" = topics where satellite and street-view are SEPARATE images
so either can be dropped cleanly: the 4 mismatch_* topics. camera_direction also
gets sat_only (drop its single query SV) but NOT sv_only (the 4 sat-arrow images
ARE the answer options — dropping them is degenerate). See reference_image_modes.

THINKING IS OFF: max_tokens is tiny (letter only); for Qwen3.5 we pass
enable_thinking=False into the chat template; Gemma4 is natively non-thinking.

Generation is BATCHED: the full (record, mode) worklist is built up front and
fed to llm.generate in one shot — vLLM schedules it across max_num_seqs.

Self-contained: reuses utils.select_images_for_record / normalize_answer and
metrics.build_full_report from the existing harness. Does NOT modify them.
"""
import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_jsonl, save_jsonl, save_json, normalize_answer, select_images_for_record
from metrics import build_full_report, roc_curves_by_topic

DATA_ROOT = Path("data")

# --- Mode applicability -----------------------------------------------------
# Topics whose satellite and street-view are SEPARATE droppable images.
MIXED_PERSPECTIVE = {
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard",
}

def modes_for_topic(topic):
    if topic in MIXED_PERSPECTIVE:
        return ["full", "sat_only", "sv_only", "blind"]
    if topic == "camera_direction":
        # sat_only = drop the single query SV (clean). sv_only degenerate (the
        # 4 sat-arrow images ARE the options). So full / sat_only / blind.
        return ["full", "sat_only", "blind"]
    return ["full", "blind"]   # single-perspective (satellite_marked) topics

# Image path classification for ablation (mirrors filter_images_for_ablation).
def is_satellite(path):
    return "/sat/" in path or "/sat_marked/" in path or "/sat_arrow/" in path
def is_streetview(path):
    return "/sv/" in path or "/composite/" in path

def filter_for_mode(images_used, mode):
    if mode == "full":
        return images_used
    if mode == "blind":
        return []
    if mode == "sat_only":
        return [im for im in images_used if not is_streetview(im["path"])]
    if mode == "sv_only":
        return [im for im in images_used if not is_satellite(im["path"])]
    raise ValueError(mode)

# --- Prompt building --------------------------------------------------------
def build_text(question, options, labels):
    valid_letters = " or ".join(sorted(options.keys()))
    image_description = "\n".join(f"Image {i + 1}: {label}" for i, label in enumerate(labels)) or "(none)"
    option_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"""You must answer this multiple-choice question.

Images provided:
{image_description}

Question: {question}

Choose the correct option based on the images.

Options:
{option_text}

IMPORTANT:
- Output ONLY ONE LETTER
- Do NOT explain
- Do NOT write words
- Only {valid_letters}

Final answer:""".strip()

def load_images(selected_images, max_px):
    pil, labels = [], []
    for item in selected_images:
        path = DATA_ROOT / item["path"]
        if not path.exists():
            print(f"Missing image: {path}", flush=True)
            continue
        img = Image.open(path).convert("RGB")
        if max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        pil.append(img)
        labels.append(item["label"])
    return pil, labels

def extract_prob_dict(logprobs_at_0, valid_letters):
    prob = {l: 0.0 for l in valid_letters}
    if not logprobs_at_0:
        return prob
    for _, lp in logprobs_at_0.items():
        tok = lp.decoded_token.strip().upper()
        if tok in prob:
            prob[tok] = max(prob[tok], math.exp(lp.logprob))
    return prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF repo id of the AWQ model")
    ap.add_argument("--model-name", required=True, help="short label for output files")
    ap.add_argument("--data", default="data/benchmark_with_answers.jsonl")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-img-px", type=int, default=1120)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="leave thinking ON (default OFF)")
    ap.add_argument("--limit-per-topic", type=int, default=0, help="smoke test cap")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="restrict to a subset of modes (debug)")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/{args.model_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.data)
    if args.limit_per_topic:
        from collections import defaultdict
        byt = defaultdict(list)
        for r in records:
            byt[r.get("topic")].append(r)
        records = [r for rs in byt.values() for r in rs[: args.limit_per_topic]]

    # Build worklist: one entry per (record, applicable mode).
    work = []
    for idx, rec in enumerate(records):
        topic = rec.get("topic", "")
        modes = modes_for_topic(topic)
        if args.modes:
            modes = [m for m in modes if m in args.modes]
        base_imgs = select_images_for_record(rec)
        for mode in modes:
            work.append((idx, rec, mode, filter_for_mode(base_imgs, mode)))
    print(f"[{args.model_name}] worklist: {len(work)} (record,mode) over {len(records)} records", flush=True)
    from collections import Counter
    print("  per-mode:", dict(Counter(w[2] for w in work)), flush=True)

    # Load model.
    print(f"Loading vLLM model: {args.model_id}", flush=True)
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 6},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    sampling = SamplingParams(temperature=0, max_tokens=8, logprobs=20)

    chat_kwargs = {}
    if not args.enable_thinking:
        # Harmless for models without the flag; turns OFF Qwen3.5 thinking.
        chat_kwargs["enable_thinking"] = False

    # Build all vLLM inputs (prompt + images) up front for batched generate.
    vinputs, meta = [], []
    for (idx, rec, mode, imgs) in work:
        pil, labels = load_images(imgs, args.max_img_px)
        text = build_text(rec["question"], rec["options"], labels)
        conversation = [{
            "role": "user",
            "content": [{"type": "image"} for _ in pil] + [{"type": "text", "text": text}],
        }]
        try:
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False, **chat_kwargs)
        except TypeError:
            # processor ignores unknown kwargs on some versions
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)
        vinputs.append({"prompt": prompt,
                        "multi_modal_data": {"image": pil} if pil else {}})
        meta.append((idx, rec, mode, [im["path"] for im in imgs]))

    print(f"Generating {len(vinputs)} prompts (batched, max_num_seqs={args.max_num_seqs})...", flush=True)
    outputs = llm.generate(vinputs, sampling_params=sampling)

    # Collect rows per mode.
    rows_by_mode = {}
    for (idx, rec, mode, used_paths), out in zip(meta, outputs):
        res = out.outputs[0]
        decoded = res.text.strip()
        options = rec["options"]
        gold = normalize_answer(rec.get("answer"), options=options)
        pred = normalize_answer(decoded, options=options)
        prob_dict = None
        if res.logprobs:
            prob_dict = extract_prob_dict(res.logprobs[0], sorted(options.keys()))
        row = {
            "index": idx,
            "question_id": rec.get("question_id"),
            "sample_id": rec.get("sample_id"),
            "question": rec["question"], "options": options,
            "gold": gold, "prediction": pred,
            "is_correct": (gold is not None and pred is not None and gold == pred),
            "topic": rec.get("topic"), "difficulty": rec.get("difficulty"),
            "land_use": rec.get("land_use"), "city": rec.get("city"),
            "country": rec.get("country"), "city_type": rec.get("city_type"),
            "benchmark_city_type": rec.get("benchmark_city_type"),
            "generation_method": rec.get("generation_method"),
            "mismatch_strategy": rec.get("mismatch_strategy"),
            "mismatch_is_match": rec.get("mismatch_is_match"),
            "query_stv_angle": rec.get("query_stv_angle"),
            "model": args.model_name, "ablation_mode": mode,
            "images_used": used_paths, "raw_response": decoded,
            "prob_dict": prob_dict,
        }
        rows_by_mode.setdefault(mode, []).append(row)

    # Per-mode outputs + a combined report index.
    summary = {"model": args.model_name, "model_id": args.model_id,
               "thinking": "on" if args.enable_thinking else "off",
               "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
               "modes": {}}
    all_rows = []
    for mode, rows in rows_by_mode.items():
        save_jsonl(rows, outdir / f"{args.model_name}_{mode}_predictions.jsonl")
        report = build_full_report(rows)
        save_json(report, outdir / f"{args.model_name}_{mode}_report.json")
        save_json(roc_curves_by_topic(rows), outdir / f"{args.model_name}_{mode}_roc_curves.json")
        summary["modes"][mode] = {"n": len(rows), "overall": report["overall"],
                                  "by_topic": {t: a["accuracy"] for t, a in report["by_topic"].items()}}
        all_rows.extend(rows)
        print(f"  [{mode}] n={len(rows)} acc={report['overall']['accuracy']:.4f}", flush=True)
    save_jsonl(all_rows, outdir / f"{args.model_name}_all_predictions.jsonl")
    save_json(summary, outdir / f"{args.model_name}_summary.json")
    print(f"\nDONE. Outputs in {outdir}/", flush=True)
    print(json.dumps({m: d["overall"] for m, d in summary["modes"].items()}, indent=2), flush=True)

if __name__ == "__main__":
    main()
