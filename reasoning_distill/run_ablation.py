#!/usr/bin/env python3
"""Full-dataset image-ablation run against the local teacher VLM (Qwen3.5-27B).

For EVERY record in train + validation + benchmark, run TWO passes:
  - full  : system + (question+options) + images   -> <think> + answer
  - blind : system + (question+options)            -> <think> + answer (no imgs)

Thinking is BUDGETED, not hard-truncated:
  - Phase 1 asks for up to THINK_BUDGET tokens.
  - If the model closes </think> on its own, we already have the answer.
  - If it hits the budget still inside <think> (spiral), Phase 2 force-closes by
    appending the partial thinking + "</think>\n" and generating only the final
    answer (ANSWER_BUDGET tokens). Guarantees a parseable letter every time.

Everything is logged per record for downstream analysis: thinking text, answer
text, parsed letter, correctness, token counts, finish reason, whether the think
budget was hit, and latency. Output is one JSONL line per (question_id, pass).

RESUMABLE: on restart, already-completed (question_id, pass) keys are skipped.

vLLM must be served with --reasoning-parser qwen3 so `reasoning_content`
(the <think> body) and `content` (the answer) come back already split.

No frontier API is touched — local vLLM only.

Usage (server up, see serve_teacher.sh):
  python run_ablation.py                      # all 3 sources, both passes
  python run_ablation.py --sources train      # just train
  python run_ablation.py --limit-per-topic 50 # smoke test
  python run_ablation.py --concurrency 32
"""
import argparse
import asyncio
import base64
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import aiohttp

ROOT = Path("/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final")

# Each source = (jsonl file, image base dir). Image paths in the JSONL are
# relative to that base dir (per-split images/).
#
# We use splits_per_city (NOT splits_seen_unseen): per-city is the random split
# with supposedly zero image leak, and the underlying records are the same data
# as seen/unseen — no point processing both.
SOURCES = {
    "train":      (ROOT / "splits_per_city" / "train.jsonl",      ROOT / "splits_per_city"),
    "validation": (ROOT / "splits_per_city" / "validation.jsonl", ROOT / "splits_per_city"),
    "benchmark":  (ROOT / "benchmark" / "benchmark_with_answers.jsonl", ROOT / "benchmark"),
}

# Mirror training/config.py so accuracy is comparable to the verdict tables,
# but ASK FOR REASONING (the letter-only instruction is replaced).
SYS_VISION = (
    "You are an urban geography expert analyzing satellite and street-level imagery. "
    "Think step by step about the images, then answer the multiple-choice question. "
    "End your reply with the single letter of the correct answer (A, B, C, or D)."
)
SYS_TEXT = (
    "You are an urban geography expert. "
    "Think step by step about the question and options, then answer. "
    "End your reply with the single letter of the correct answer (A, B, C, or D)."
)

THINK_BUDGET = 12000   # tokens allowed inside <think> before we force-close
ANSWER_BUDGET = 64     # tokens for the forced final answer in phase 2
TEMPERATURE = 0.6      # model's thinking-mode default (generation_config.json)
TOP_P = 0.95
TOP_K = 20


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def b64_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def image_paths_for(record, base_dir):
    imgs = record.get("images", {})
    out = []
    if imgs.get("satellite"):
        out.append(base_dir / imgs["satellite"])
    for k in ("along_fwd", "along_bwd", "cross_left", "cross_right"):
        p = imgs.get(f"streetview_{k}")
        if isinstance(p, str) and p.endswith(".jpg"):
            out.append(base_dir / p)
    return [p for p in out if p.exists()]


def question_text(record):
    opts = record["options"]
    return record["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())


def build_messages(record, base_dir, with_images):
    content = [{"type": "text", "text": question_text(record)}]
    if with_images:
        for p in image_paths_for(record, base_dir):
            content.append({"type": "image_url", "image_url": {"url": b64_image(p)}})
    sys_prompt = SYS_VISION if with_images else SYS_TEXT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def parse_letter(text):
    """Robust answer-letter extraction (ignores letters inside words)."""
    import re
    if not text:
        return None
    t = text.strip()
    m = re.fullmatch(r"\(?([ABCD])\)?[.):]?", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    # last standalone A-D token wins (the final commitment)
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", t)
    return matches[-1].upper() if matches else None


async def chat(session, base_url, messages, max_tokens, **extra):
    payload = {
        "model": "teacher", "messages": messages,
        "max_tokens": max_tokens, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K, **extra,
    }
    async with session.post(f"{base_url}/v1/chat/completions", json=payload) as r:
        r.raise_for_status()
        data = await r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "reasoning": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


async def run_one(session, base_url, record, base_dir, with_images):
    """One pass (full or blind) with budgeted thinking + forced-close fallback."""
    t0 = time.monotonic()
    messages = build_messages(record, base_dir, with_images)
    r1 = await chat(session, base_url, messages, THINK_BUDGET + ANSWER_BUDGET)

    reasoning, answer_text = r1["reasoning"], r1["content"]
    think_truncated = False

    # Spiral case: budget hit while still inside <think> (parser put everything in
    # reasoning_content, content empty, finish_reason=length). Force-close.
    if not answer_text.strip() and r1["finish_reason"] == "length":
        think_truncated = True
        forced = messages + [{
            "role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n\n",
        }]
        # continue_final_message: append generation to the assistant turn we just
        # supplied (the force-closed think block) instead of starting a new turn.
        # These are top-level chat-request fields in vLLM, not extra_body.
        r2 = await chat(session, base_url, forced, ANSWER_BUDGET,
                        add_generation_prompt=False,
                        continue_final_message=True)
        answer_text = r2["content"] or r2["reasoning"]
        r1["usage"] = {k: r1["usage"].get(k, 0) + r2["usage"].get(k, 0)
                       for k in set(r1["usage"]) | set(r2["usage"])}

    letter = parse_letter(answer_text) or parse_letter(reasoning)
    return {
        "reasoning": reasoning,
        "answer_text": answer_text,
        "letter": letter,
        "finish_reason": r1["finish_reason"],
        "think_truncated": think_truncated,
        "usage": r1["usage"],
        "latency_s": round(time.monotonic() - t0, 2),
    }


async def worker(name, queue, session, base_url, out_f, lock, counters):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        source, record = item
        base_dir = SOURCES[source][1]
        gt = record.get("answer")
        try:
            for with_images, passname in ((True, "full"), (False, "blind")):
                res = await run_one(session, base_url, record, base_dir, with_images)
                correct = res["letter"] == gt
                row = {
                    "question_id": record["question_id"],
                    "sample_id": record.get("sample_id"),
                    "source": source, "pass": passname,
                    "topic": record["topic"], "city": record.get("city"),
                    "image_mode": record.get("image_mode"),
                    "gt": gt, "letter": res["letter"], "correct": correct,
                    "reasoning": res["reasoning"], "answer_text": res["answer_text"],
                    "think_truncated": res["think_truncated"],
                    "finish_reason": res["finish_reason"],
                    "usage": res["usage"], "latency_s": res["latency_s"],
                }
                async with lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    counters["done"] += 1
                    counters[f"{passname}_correct"] += int(correct)
                    if res["think_truncated"]:
                        counters["truncated"] += 1
        except Exception as e:  # one bad record shouldn't kill the run
            async with lock:
                out_f.write(json.dumps({
                    "question_id": record["question_id"], "source": source,
                    "error": repr(e),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["errors"] += 1
        finally:
            queue.task_done()
            if counters["done"] % 50 == 0:
                d = counters["done"]
                fc = counters["full_correct"]
                bc = counters["blind_correct"]
                done_recs = d // 2 or 1
                print(f"  {d} passes | full {100*fc/done_recs:.0f}% "
                      f"blind {100*bc/done_recs:.0f}% | "
                      f"trunc {counters['truncated']} err {counters['errors']}",
                      flush=True)


def load_done_keys(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "pass" in r and "question_id" in r:
                done.add((r["question_id"], r["pass"]))
    return done


async def main_async(args):
    # Build worklist
    work = []
    for source in args.sources:
        path, _ = SOURCES[source]
        recs = load_jsonl(path)
        if args.limit_per_topic:
            by_t = defaultdict(list)
            for r in recs:
                by_t[r["topic"]].append(r)
            recs = [r for rs in by_t.values() for r in rs[: args.limit_per_topic]]
        for r in recs:
            work.append((source, r))

    # Resume: skip records whose BOTH passes are already logged
    done_keys = load_done_keys(args.out)
    if done_keys:
        before = len(work)
        work = [(s, r) for (s, r) in work
                if not ((r["question_id"], "full") in done_keys
                        and (r["question_id"], "blind") in done_keys)]
        print(f"Resume: {before - len(work)} records already complete, "
              f"{len(work)} remain.", flush=True)

    print(f"Running {len(work)} records x 2 passes = {2*len(work)} generations "
          f"across {args.sources} @ concurrency {args.concurrency}", flush=True)

    counters = defaultdict(int)
    queue = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    out_f = open(args.out, "a")
    lock = asyncio.Lock()
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        workers = [
            asyncio.create_task(
                worker(f"w{i}", queue, session, args.base_url, out_f, lock, counters))
            for i in range(args.concurrency)
        ]
        await queue.join()
        await asyncio.gather(*workers)
    out_f.close()

    print("\n=== DONE ===", flush=True)
    print(f"passes: {counters['done']} | truncated-think: {counters['truncated']} "
          f"| errors: {counters['errors']}")
    print(f"log: {args.out}")
    print("Run analyze.py for the per-topic full-vs-blind table.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--sources", nargs="+",
                    default=["train", "validation", "benchmark"],
                    choices=list(SOURCES))
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit-per-topic", type=int, default=0,
                    help="cap records per topic per source (smoke testing)")
    ap.add_argument("--out", default="reasoning_distill/ablation_log.jsonl")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
