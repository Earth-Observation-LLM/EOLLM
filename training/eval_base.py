"""
eval_base.py — Fast batched eval of Qwen3.5-4B-VL on the full validation set.

Optimizations vs the training-time callback eval:
  - DataLoader with num_workers=8 for parallel image prep
  - Samples sorted by image count to minimize padding waste
  - Greedy decoding (deterministic, MCQ doesn't need sampling)
  - High max_new_tokens cap (128) + EOS stopping
  - Lenient letter parse: first A/B/C/D in output

Usage:
    conda activate unsloth
    python training/eval_base.py                   # full raw base
    EVAL_ADAPTER=/path/to/adapter python ...       # eval a LoRA adapter
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import Dataset, DataLoader

from config import SEED, SPLIT, SYSTEM_PROMPT, find_dataset_dir, detect_profile, SCRIPT_DIR
from data import load_jsonl, convert_record


BATCH_SIZE = 32
MAX_NEW_TOKENS = 16  # letter + punctuation is enough; don't waste GPU on rambling
NUM_WORKERS = 8
LETTER_RE_STRICT = re.compile(r"\b([ABCD])\b")
# Fallback: letter with non-word char on BOTH sides (or string boundaries).
# Avoids false positives on words like "Based" (where D is at end of word) or
# "By" (where B is at start of word). Word-boundary via explicit lookaround so
# we don't double-match inside the strict regex's territory.
LETTER_RE_LOOSE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?=[\s\.\):,!?]|$)")


def _pad_id(tokenizer):
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    pid = getattr(inner, "pad_token_id", None)
    if pid is None:
        pid = getattr(inner, "eos_token_id", None)
    return pid


def parse_letter(text: str) -> str | None:
    """Extract isolated A/B/C/D letter from model output.

    Strategy:
      1. Strict word-boundary match (most common — bare letter or "A.", " A ").
      2. Fallback: letter followed by punctuation/space/eol (catches "A:" etc.).
      3. Return None if neither matches (scored wrong, not silently leniently matched).
    """
    upper = text.upper()
    m = LETTER_RE_STRICT.search(upper)
    if m:
        return m.group(1)
    m = LETTER_RE_LOOSE.search(upper)
    return m.group(1) if m else None


class PreppedSamples(Dataset):
    """Worker-parallel preprocessing of records to (text, images, meta) tuples."""

    def __init__(self, records: list[dict], base_dir: str, max_edge: int, tokenizer):
        self.records = records
        self.base_dir = base_dir
        self.max_edge = max_edge
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        converted = convert_record(rec, self.base_dir, self.max_edge)
        user_msg = converted["messages"][1]
        images = [p["image"] for p in user_msg["content"] if p["type"] == "image"]
        inf_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                p if p["type"] == "text" else {"type": "image"}
                for p in user_msg["content"]
            ]},
        ]
        text = self.tokenizer.apply_chat_template(inf_messages, add_generation_prompt=True, tokenize=False)
        return {
            "text": text,
            "images": images,
            "answer": rec["answer"],
            "topic": rec["topic"],
            "difficulty": rec.get("difficulty", "unknown"),
            "question_id": rec["question_id"],
            "n_images": len(images),
        }


def collate(batch):
    return batch  # keep list-of-dicts; we handle tokenization in the main thread (uses GPU-bound tokenizer)


def eval_batched(model, tokenizer, records, base_dir, max_edge):
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)

    # Sort by image_mode (proxy for image count) to pack similar-length batches
    mode_order = {
        "satellite_only": 0, "streetview_only": 1,
        "streetview_mega": 2,  # 2 images (sat_marked + mega composite)
        "satellite_arrow": 3,  # 4 images
        "streetview_composite": 4, "streetview_binary": 4,  # 5 images
    }
    prepped = PreppedSamples(records, base_dir, max_edge, tokenizer)
    sorted_indices = sorted(
        range(len(prepped)),
        key=lambda i: mode_order.get(records[i].get("image_mode", "satellite_only"), 5),
    )

    dataset = torch.utils.data.Subset(prepped, sorted_indices)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=False,  # images are PIL, not tensors
    )

    topic_results: dict[str, list[bool]] = defaultdict(list)
    diff_results: dict[str, list[bool]] = defaultdict(list)
    pad_id = _pad_id(tokenizer)

    n_total = len(records)
    t0 = time.time()
    done = 0

    for batch in loader:
        texts = [item["text"] for item in batch]
        images_list = [item["images"] for item in batch]

        inputs = tokenizer(
            images_list, texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                do_sample=False,  # greedy — deterministic and faster for MCQ
                pad_token_id=pad_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i, item in enumerate(batch):
            gen_tokens = output_ids[i][input_len:]
            generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            letter = parse_letter(generated)
            correct = letter == item["answer"]
            topic_results[item["topic"]].append(correct)
            diff_results[item["difficulty"]].append(correct)

        done += len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_total - done) / rate if rate > 0 else 0
        running_correct = sum(sum(v) for v in topic_results.values())
        running_total = sum(len(v) for v in topic_results.values())
        print(f"  [{done}/{n_total}] acc={running_correct/running_total:.1%} "
              f"rate={rate:.1f} s/s eta={eta/60:.1f}min elapsed={elapsed/60:.1f}min",
              flush=True)

    per_topic = {}
    total_correct = total_n = 0
    for topic, results in sorted(topic_results.items()):
        c = sum(results); n = len(results)
        per_topic[topic] = {"acc": c / n if n else 0, "correct": c, "n": n}
        total_correct += c; total_n += n

    per_difficulty = {}
    for diff, results in sorted(diff_results.items()):
        c = sum(results); n = len(results)
        per_difficulty[diff] = {"acc": c / n if n else 0, "correct": c, "n": n}

    return {
        "overall": total_correct / total_n if total_n else 0,
        "per_topic": per_topic,
        "per_difficulty": per_difficulty,
        "n_total": total_n,
        "n_correct": total_correct,
    }


def main():
    t_start = time.time()
    profile_name, CFG = detect_profile()

    adapter_path = os.environ.get("EVAL_ADAPTER")
    tag = Path(adapter_path).name if adapter_path else "base"
    out_dir = SCRIPT_DIR / "runs" / f"{tag}_eval_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = find_dataset_dir()
    split_dir = dataset_dir / SPLIT
    val_records = load_jsonl(str(split_dir / "validation.jsonl"))
    print(f"Val records: {len(val_records)}")
    print(f"Profile: {profile_name}, image_max_edge={CFG['image_max_edge']}, "
          f"bs={BATCH_SIZE}, workers={NUM_WORKERS}, max_new_tokens={MAX_NEW_TOKENS}")

    local_model_path = str(SCRIPT_DIR.parent / "models" / "Qwen3.5-4B")
    model_name = local_model_path if Path(local_model_path).exists() else "unsloth/Qwen3.5-4B"
    print(f"Model: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8192,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adapter loaded.")

    # Left padding for decoder-only generation
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is not None and hasattr(obj, "padding_side"):
            obj.padding_side = "left"

    print("Running batched full val eval...")

    t_eval = time.time()
    accuracy = eval_batched(model, tokenizer, val_records, str(split_dir), CFG["image_max_edge"])
    elapsed = time.time() - t_eval
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Overall: {accuracy['overall']:.1%} ({accuracy['n_correct']}/{accuracy['n_total']})")
    for topic, info in sorted(accuracy["per_topic"].items()):
        print(f"  {topic}: {info['acc']:.1%} ({info['correct']}/{info['n']})")

    lines = [
        f"# {tag} — Full Val Eval\n",
        f"**Model:** {model_name}" + (f" + adapter {adapter_path}" if adapter_path else " (NO LoRA)") + "\n",
        f"**Samples:** {accuracy['n_total']} (full validation set)\n",
        f"**Config:** bs={BATCH_SIZE}, workers={NUM_WORKERS}, max_new_tokens={MAX_NEW_TOKENS}, greedy, lenient-parse\n",
        f"**Eval time:** {elapsed:.0f}s ({elapsed/60:.1f} min)\n",
        f"\n## Overall: {accuracy['overall']:.1%} ({accuracy['n_correct']}/{accuracy['n_total']})\n",
        "\n## Per-Topic Accuracy\n",
        "| Topic | Accuracy | Correct/Total |",
        "|-------|----------|---------------|",
    ]
    for topic, info in sorted(accuracy["per_topic"].items()):
        lines.append(f"| {topic} | {info['acc']:.1%} | {info['correct']}/{info['n']} |")

    if accuracy.get("per_difficulty"):
        lines.append("\n## Per-Difficulty Accuracy\n")
        lines.append("| Difficulty | Accuracy | Correct/Total |")
        lines.append("|------------|----------|---------------|")
        for diff, info in sorted(accuracy["per_difficulty"].items()):
            lines.append(f"| {diff} | {info['acc']:.1%} | {info['correct']}/{info['n']} |")

    verdict_path = out_dir / f"{tag}_verdict.md"
    verdict_path.write_text("\n".join(lines))

    with open(out_dir / f"{tag}_accuracy.json", "w") as f:
        json.dump(accuracy, f, indent=2)

    print(f"\nVerdict: {verdict_path}")
    print(f"Total wall: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
