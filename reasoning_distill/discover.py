#!/usr/bin/env python3
"""Dataset discovery + image-ablation gate against the local teacher VLM.

For a stratified sample of records, run TWO passes through the served model:
  - full  : system + (question+options) + images   -> answer
  - blind : system + (question+options)            -> answer  (no images)

Then bucket each record by (full_correct, blind_correct):
  visual    full ✓ / blind ✗  -> genuinely needs vision (gold for distillation)
  leaked    full ✓ / blind ✓  -> solvable without images (flag)
  hard      full ✗ / blind ✗  -> teacher can't do it either
  regress   full ✗ / blind ✓  -> images HURT (interesting)

Per-topic aggregates are the benchmark signal the report wants: teacher
accuracy with vs without vision. NO frontier API is touched — local vLLM only.

The full/blind requests share the identical text prefix, so with
--enable-prefix-caching the blind pass is nearly free after the full pass.

Usage (server must be up, see serve_teacher.sh):
  python discover.py --n-per-topic 20
  python discover.py --n-per-topic 20 --topics transit_density amenity_richness
"""
import argparse, base64, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path("/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final")
SPLIT_DIR = ROOT / "splits_seen_unseen"
TRAIN = SPLIT_DIR / "train.jsonl"
# Image paths in the JSONL (e.g. "images/sat/foo.png") are relative to the
# SPLIT directory, not the dataset root — images live per-split.
BASE_DIR = SPLIT_DIR

# Mirror training/config.py system prompts so our numbers are comparable to the
# existing verdict tables.
SYS_VISION = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Answer the multiple-choice question based on the provided images. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)
SYS_TEXT = (
    "You are an urban geography expert. "
    "Answer the multiple-choice question based on the question and options text. "
    "Reply with only the letter of the correct answer (A, B, C, or D)."
)

SEED = 3407  # match training seed


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def b64_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def image_paths_for(record):
    """The image files this record's image_mode would feed the model.

    Mirrors training/data.py: satellite_marked/only -> 1 sat image; the SV-heavy
    modes add the 4 cardinal SV images. We keep it simple here (discovery, not
    training) and just attach whatever files the record references.
    """
    imgs = record.get("images", {})
    out = []
    sat = imgs.get("satellite")
    if sat:
        out.append(BASE_DIR / sat)
    for k in ("along_fwd", "along_bwd", "cross_left", "cross_right"):
        p = imgs.get(f"streetview_{k}")
        if isinstance(p, str) and p.endswith(".jpg"):
            out.append(BASE_DIR / p)
    return [p for p in out if p.exists()]


def question_text(record):
    opts = record["options"]
    return record["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())


def build_messages(record, with_images):
    qt = question_text(record)
    content = [{"type": "text", "text": qt}]
    if with_images:
        for p in image_paths_for(record):
            content.append({"type": "image_url", "image_url": {"url": b64_image(p)}})
    sys_prompt = SYS_VISION if with_images else SYS_TEXT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def parse_letter(text):
    """Extract the answer letter, ignoring letters embedded in words.

    The model may reply "C", "C.", "Answer: B", or (with reasoning) a sentence
    ending in a choice. Strategy: prefer a standalone A-D token; fall back to a
    letter immediately followed by ')' or '.'; never match an 'a' inside a word.
    """
    import re
    t = text.strip()
    # 1) the whole reply is just a letter (optionally punctuated): "C", "C."
    m = re.fullmatch(r"\(?([ABCD])\)?[.):]?", t, re.I)
    if m:
        return m.group(1).upper()
    # 2) "Answer: X" / "answer is X" near the end
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    # 3) a standalone letter token anywhere (word-boundaried, not inside a word)
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", t)
    if m:
        return m.group(1).upper()
    return None


def call(base_url, record, with_images, max_tokens):
    msgs = build_messages(record, with_images)
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"model": "teacher", "messages": msgs,
              "max_tokens": max_tokens, "temperature": 0.0},
        timeout=300,
    )
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    return parse_letter(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n-per-topic", type=int, default=20)
    ap.add_argument("--topics", nargs="*", default=None)
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="discovery uses letter-only; bump for reasoning traces")
    ap.add_argument("--out", default="reasoning_distill/discovery_log.jsonl")
    args = ap.parse_args()

    rng = random.Random(SEED)
    records = load_jsonl(TRAIN)
    by_topic = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r)

    topics = args.topics or sorted(by_topic)
    sample = []
    for t in topics:
        pool = by_topic[t]
        rng.shuffle(pool)
        sample.extend(pool[: args.n_per_topic])

    print(f"Probing {len(sample)} records across {len(topics)} topics "
          f"({args.n_per_topic}/topic)...", file=sys.stderr)

    stats = defaultdict(lambda: defaultdict(int))  # topic -> bucket -> n
    logf = open(args.out, "w")
    for i, rec in enumerate(sample, 1):
        gt = rec["answer"]
        f_ans, f_raw = call(args.base_url, rec, True, args.max_tokens)
        b_ans, b_raw = call(args.base_url, rec, False, args.max_tokens)
        fc, bc = f_ans == gt, b_ans == gt
        bucket = ("visual" if fc and not bc else "leaked" if fc and bc
                  else "regress" if not fc and bc else "hard")
        t = rec["topic"]
        stats[t][bucket] += 1
        stats[t]["full_correct"] += fc
        stats[t]["blind_correct"] += bc
        stats[t]["n"] += 1
        logf.write(json.dumps({
            "question_id": rec["question_id"], "topic": t, "gt": gt,
            "full_ans": f_ans, "blind_ans": b_ans, "bucket": bucket,
        }) + "\n")
        if i % 20 == 0:
            print(f"  {i}/{len(sample)}", file=sys.stderr)
    logf.close()

    # Report
    print(f"\n{'topic':22s} {'n':>4} {'full%':>6} {'blind%':>7} "
          f"{'visual':>7} {'leaked':>7} {'hard':>5} {'regr':>5}")
    print("-" * 72)
    for t in topics:
        s = stats[t]
        n = s["n"] or 1
        print(f"{t:22s} {s['n']:>4} "
              f"{100*s['full_correct']/n:>5.0f}% {100*s['blind_correct']/n:>6.0f}% "
              f"{s['visual']:>7} {s['leaked']:>7} {s['hard']:>5} {s['regress']:>5}")
    print(f"\nfull%-blind% gap = visual dependency. Log: {args.out}")


if __name__ == "__main__":
    main()
