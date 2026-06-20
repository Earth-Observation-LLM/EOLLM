"""
evaluation.py — Inference, accuracy metrics, and eval reporting.

Shares generation settings and letter-parsing with eval_base.py so mid-training
eval and external base/adapter evals produce comparable numbers. No sampling.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch

from config import SEED, SYSTEM_PROMPT
from data import convert_record
from eval_base import parse_letter, _pad_id


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _set_left_padding(tokenizer) -> None:
    """Decoder-only generation needs left-padding for correct attention over
    the prompt. Processor wraps an inner tokenizer; set both if present."""
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is not None and hasattr(obj, "padding_side"):
            obj.padding_side = "left"


def _generate_letter(model, tokenizer, inputs) -> str | None:
    """Greedy 16-token decode; return parsed letter or None."""
    pad_id = _pad_id(tokenizer)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            use_cache=True,
            do_sample=False,
            pad_token_id=pad_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return generated, parse_letter(generated)


def _prepare_inference_inputs(rec: dict, base_dir: str, max_edge: int, tokenizer):
    """Build tokenizer inputs for a single inference sample. Returns (inputs, images)."""
    converted = convert_record(rec, base_dir, max_edge)
    user_msg = converted["messages"][1]
    images = [p["image"] for p in user_msg["content"] if p["type"] == "image"]

    inf_messages = [
        # System content as a plain STRING (not list-of-parts) to match training
        # (data.py convert_record). Qwen renders both identically, but Gemma's
        # template appends a trailing space to list-of-parts items — a train/eval
        # mismatch that only surfaces for Gemma. String form matches both.
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            p if p["type"] == "text" else {"type": "image"}
            for p in user_msg["content"]
        ]},
    ]

    # enable_thinking=False: Qwen3.5's template otherwise opens a <think> block at
    # the assistant turn, so the model reasons for many tokens BEFORE emitting the
    # answer letter. With a small max_new_tokens that reasoning gets truncated and
    # no letter is ever produced -> parse_letter returns None -> ~0% accuracy. The
    # model is TRAINED to answer with a bare letter, so suppress thinking at eval to
    # match the training target. (Diagnosed: think-on/16tok=0%, think-off/16tok=75%.)
    text = tokenizer.apply_chat_template(
        inf_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    if not images:
        inputs = tokenizer(text=text, add_special_tokens=False, return_tensors="pt").to("cuda")
    else:
        img_input = images[0] if len(images) == 1 else images
        inputs = tokenizer(img_input, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs


# ---------------------------------------------------------------------------
# Run inference on a few samples
# ---------------------------------------------------------------------------


def run_eval_samples(
    model,
    tokenizer,
    records: list[dict],
    base_dir: str,
    max_edge: int,
    n: int = 3,
    seed: int = SEED + 1,
) -> list[dict]:
    """Run inference on n random validation samples. Greedy; shared parser."""
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)
    _set_left_padding(tokenizer)

    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), min(n, len(records)))
    results = []

    for idx in indices:
        rec = records[idx]
        inputs = _prepare_inference_inputs(rec, base_dir, max_edge, tokenizer)
        generated, letter = _generate_letter(model, tokenizer, inputs)

        results.append({
            "question_id": rec["question_id"],
            "topic": rec["topic"],
            "question": rec["question"],
            "gold": rec["answer"],
            "predicted": generated,
            "letter": letter,
            "correct": letter == rec["answer"],
        })

    return results


# ---------------------------------------------------------------------------
# Per-topic accuracy on a larger sample
# ---------------------------------------------------------------------------


def compute_topic_accuracy(
    model,
    tokenizer,
    records: list[dict],
    base_dir: str,
    max_edge: int,
    n: int = 300,
    seed: int = SEED + 2,
) -> dict:
    """Compute accuracy grouped by topic on n random validation samples.

    Greedy decoding; shared parse_letter. Deterministic for a given (seed, n).
    """
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)
    _set_left_padding(tokenizer)

    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), min(n, len(records)))

    topic_results: dict[str, list[bool]] = defaultdict(list)
    diff_results: dict[str, list[bool]] = defaultdict(list)

    for idx in indices:
        rec = records[idx]
        inputs = _prepare_inference_inputs(rec, base_dir, max_edge, tokenizer)
        _, letter = _generate_letter(model, tokenizer, inputs)
        correct = letter == rec["answer"]
        topic_results[rec["topic"]].append(correct)
        difficulty = rec.get("difficulty", "unknown")
        diff_results[difficulty].append(correct)

    per_topic = {}
    total_correct = 0
    total_n = 0
    for topic, results in sorted(topic_results.items()):
        n_correct = sum(results)
        n_total = len(results)
        per_topic[topic] = {"acc": n_correct / n_total if n_total else 0, "correct": n_correct, "n": n_total}
        total_correct += n_correct
        total_n += n_total

    per_difficulty = {}
    for diff, results in sorted(diff_results.items()):
        n_correct = sum(results)
        n_total = len(results)
        per_difficulty[diff] = {"acc": n_correct / n_total if n_total else 0, "correct": n_correct, "n": n_total}

    return {
        "overall": total_correct / total_n if total_n else 0,
        "per_topic": per_topic,
        "per_difficulty": per_difficulty,
        "n_total": total_n,
        "n_correct": total_correct,
    }


# ---------------------------------------------------------------------------
# Write reports
# ---------------------------------------------------------------------------


def write_eval_md(base_results: list[dict], ft_results: list[dict], path: str):
    """Write eval_samples.md comparing base vs finetuned."""
    lines = ["# Eval Samples — Base vs Finetuned\n"]
    for b, f in zip(base_results, ft_results):
        lines.append(f"## {b['question_id']} ({b['topic']})\n")
        lines.append(f"**Question:** {b['question']}\n")
        lines.append(f"**Gold answer:** {b['gold']}\n")
        lines.append("| Model | Predicted | Correct |")
        lines.append("|-------|-----------|---------|")
        lines.append(f"| Base | {b['predicted'][:80]} | {'yes' if b['correct'] else 'no'} |")
        lines.append(f"| Finetuned | {f['predicted'][:80]} | {'yes' if f['correct'] else 'no'} |")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_accuracy_md(accuracy: dict, path: str):
    """Write per-topic accuracy report."""
    lines = [
        "# Per-Topic Accuracy\n",
        f"**Overall: {accuracy['overall']:.1%}** ({accuracy['n_correct']}/{accuracy['n_total']})\n",
        "| Topic | Accuracy | Correct | Total |",
        "|-------|----------|---------|-------|",
    ]
    for topic, info in accuracy["per_topic"].items():
        lines.append(f"| {topic} | {info['acc']:.1%} | {info['correct']} | {info['n']} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
