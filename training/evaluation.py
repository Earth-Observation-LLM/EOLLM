"""
evaluation.py — Inference, accuracy metrics, and eval reporting.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch

from config import SEED, SYSTEM_PROMPT
from data import convert_record


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
) -> list[dict]:
    """Run inference on n random validation samples."""
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    rng = random.Random(SEED + 1)
    indices = rng.sample(range(len(records)), min(n, len(records)))
    results = []

    for idx in indices:
        rec = records[idx]
        converted = convert_record(rec, base_dir, max_edge)

        user_msg = converted["messages"][1]
        images = [p["image"] for p in user_msg["content"] if p["type"] == "image"]

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
                **inputs, max_new_tokens=32, use_cache=True,
                temperature=0.7, top_p=0.8, top_k=20,
            )

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

    Returns {"overall": float, "per_topic": {topic: {"acc": float, "n": int, "correct": int}}, "n_total": int}
    """
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), min(n, len(records)))

    topic_results: dict[str, list[bool]] = defaultdict(list)

    for idx in indices:
        rec = records[idx]
        converted = convert_record(rec, base_dir, max_edge)

        user_msg = converted["messages"][1]
        images = [p["image"] for p in user_msg["content"] if p["type"] == "image"]

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
                **inputs, max_new_tokens=32, use_cache=True,
                temperature=0.7, top_p=0.8, top_k=20,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        correct = generated.startswith(rec["answer"])
        topic_results[rec["topic"]].append(correct)

    per_topic = {}
    total_correct = 0
    total_n = 0
    for topic, results in sorted(topic_results.items()):
        n_correct = sum(results)
        n_total = len(results)
        per_topic[topic] = {"acc": n_correct / n_total if n_total else 0, "correct": n_correct, "n": n_total}
        total_correct += n_correct
        total_n += n_total

    return {
        "overall": total_correct / total_n if total_n else 0,
        "per_topic": per_topic,
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
