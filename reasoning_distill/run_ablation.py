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

import io
import sys

import aiohttp

ROOT = Path("/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final")

# Reuse the dataset's OWN image builders so every image we send is pixel-identical
# to what training/eval feeds the model (mega tiling, arrows, corner labels, dot).
sys.path.insert(0, str(ROOT))
import composite_utils as cu  # noqa: E402

# Mirror training/data.py's corner-label burn-in (V1: small white box, top-left).
# data.py imports add_corner_label from its own module; we replicate the call by
# importing it from training/data.py to stay byte-identical.
sys.path.insert(0, "/home/ezel/Development/EOLLM/training")
try:
    from data import add_corner_label  # noqa: E402
except Exception:  # pragma: no cover - fall back to no label if import fails
    def add_corner_label(img, letter):
        return img

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

# Per-mode system prompts. Each is HONEST about the inputs actually shown, so
# the model never claims a perspective it can't see (avoids confabulation).
# We tell it to be concise (Qwen3.5 over-thinks) and to emit a strict JSON
# answer-only object after thinking, so parsing is unambiguous.
_THINK_TAIL = (
    "Reason concisely — do not over-think; reach a decision efficiently. "
    "After your reasoning, output ONLY a JSON object on its own line: "
    '{"answer": "X"} where X is one of A, B, C, or D.')
SYS_PROMPTS = {
    "full":     "You are an urban geography expert analyzing satellite and street-level imagery. " + _THINK_TAIL,
    "sat_only": "You are an urban geography expert analyzing satellite imagery. " + _THINK_TAIL,
    "sv_only":  "You are an urban geography expert analyzing street-level imagery. " + _THINK_TAIL,
    "blind":    "You are an urban geography expert. " + _THINK_TAIL,
}

THINK_BUDGET = 4000    # tokens allowed inside <think> before we force-close.
                       # Qwen3.5 over-thinks hard (median ~5k tok, spirals to
                       # 8k+); 4k roughly halves wall-clock for the 105k-gen run.
                       # Spilled-over thinking is force-closed into a JSON answer.
ANSWER_BUDGET = 64     # tokens for the forced final answer in phase 2
# temperature 0 = greedy decoding for full determinism / reproducible traces.
# (top_p/top_k are neutralized to greedy-equivalent so they don't interfere.)
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def pil_to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def perspective_images(record, base_dir):
    """Return (sat_imgs, sv_imgs) as two lists of PIL images, built with the
    dataset's OWN helpers so they're pixel-identical to training/eval.

    sat_imgs = the satellite-perspective image(s); sv_imgs = street-level
    image(s). Mode filtering (sat_only/sv_only) just picks which list(s) to send.
    See reference_image_modes memory for the exact composition per image_mode.
    """
    mode = record.get("image_mode", "satellite_only")
    base = str(base_dir)
    res = cu.get_images_for_question(record, base_dir=base)
    sat, sv = [], []

    if mode in ("satellite_only", "satellite_marked"):
        sat = [res["primary"]]                       # 1 sat, no sv

    elif mode == "satellite_arrow":
        # 4 arrow-on-satellite option images (the A/B/C/D choices) + 1 query SV.
        for letter in ("A", "B", "C", "D"):
            if res.get("options", {}).get(letter) is not None:
                sat.append(res["options"][letter])
        if res.get("query_sv") is not None:
            sv = [res["query_sv"]]

    elif mode in ("streetview_composite", "streetview_binary"):
        # 1 sat_marked + 4 separate corner-labeled SV images (mirror data.py).
        imgs = record["images"]
        sat = [cu.make_sat_marked(os.path.join(base, imgs["satellite"]))]
        if mode == "streetview_composite":
            sv_paths = [os.path.join(base, imgs[f"streetview_{a}"]) for a in cu.STV_ANGLES]
        else:
            neg = record.get("mismatch_negative_stv_paths") or [
                os.path.join(base, imgs[f"streetview_{a}"]) for a in cu.STV_ANGLES]
            sv_paths = [os.path.join(base, p) if not os.path.isabs(p) else p for p in neg]
        for path, letter in zip(sv_paths, ("A", "B", "C", "D")):
            sv.append(add_corner_label(cu.Image.open(path), letter))

    elif mode == "streetview_mega":
        # 1 sat_marked + 1 mega composite (the 16 SV frames live inside the mega).
        imgs = record["images"]
        sat = [cu.make_sat_marked(os.path.join(base, imgs["satellite"]))]
        sv = [res["primary"]]

    else:  # unknown -> whatever primary is, treated as satellite
        if res.get("primary") is not None:
            sat = [res["primary"]]
    return sat, sv


def question_text(record):
    opts = record["options"]
    return record["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts.items())


def build_messages(record, base_dir, mode):
    """Build the chat messages for a given ablation mode.

    full -> sat + sv ; sat_only -> sat ; sv_only -> sv ; blind -> no images.
    """
    content = [{"type": "text", "text": question_text(record)}]
    if mode != "blind":
        sat, sv = perspective_images(record, base_dir)
        chosen = []
        if mode in ("full", "sat_only"):
            chosen += sat
        if mode in ("full", "sv_only"):
            chosen += sv
        for img in chosen:
            content.append({"type": "image_url",
                            "image_url": {"url": pil_to_b64(img)}})
    return [
        {"role": "system", "content": SYS_PROMPTS[mode]},
        {"role": "user", "content": content},
    ]


# Which ablation modes apply to each topic. satellite_marked topics: only
# full/blind (sat_only==full, no sv). camera_direction: no sv_only (arrows ARE
# the options). mismatch_* : all four. See reference_image_modes memory.
BOTH_PERSPECTIVE_FULL = {  # full / sat_only / sv_only / blind
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard",
}
CAMERA = {"camera_direction"}  # full / sat_only / blind (no sv_only)


def modes_for_topic(topic):
    if topic in BOTH_PERSPECTIVE_FULL:
        return ["full", "sat_only", "sv_only", "blind"]
    if topic in CAMERA:
        return ["full", "sat_only", "blind"]
    return ["full", "blind"]  # the 8 satellite_marked topics


def parse_letter(text):
    """Robust answer-letter extraction. Prefers the {"answer":"X"} JSON we ask
    for; falls back to plain-text heuristics (ignoring letters inside words)."""
    import re
    if not text:
        return None
    t = text.strip()
    # 1) JSON answer object (what the prompt requests) — take the LAST one.
    js = re.findall(r'"answer"\s*:\s*"?\(?([ABCD])\)?"?', t, re.I)
    if js:
        return js[-1].upper()
    # 2) whole reply is just a letter: "C", "C."
    m = re.fullmatch(r"\(?([ABCD])\)?[.):]?", t, re.I)
    if m:
        return m.group(1).upper()
    # 3) "Answer: X" near the end
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, re.I)
    if m:
        return m.group(1).upper()
    # 4) last standalone A-D token wins (the final commitment)
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", t)
    return matches[-1].upper() if matches else None


class ContextOverflow(Exception):
    """A 400 from the server, almost always max-context exceeded."""


async def chat(session, base_url, messages, max_tokens, _attempt=0, **extra):
    payload = {
        "model": "teacher", "messages": messages,
        "max_tokens": max_tokens, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K, **extra,
    }
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as r:
            if r.status == 400:
                # context overflow / bad request — not retryable as-is
                raise ContextOverflow(await r.text())
            r.raise_for_status()
            data = await r.json()
    except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError) as e:
        # transient — exponential backoff up to 4 tries
        if _attempt >= 4:
            raise
        await asyncio.sleep(2 ** _attempt)
        return await chat(session, base_url, messages, max_tokens,
                          _attempt=_attempt + 1, **extra)
    choice = data["choices"][0]
    msg = choice["message"]
    # This vLLM build returns the <think> body under "reasoning" (not
    # "reasoning_content"); keep the fallback for portability.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "reasoning": reasoning,
        "content": msg.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}) or {},
    }


async def run_one(session, base_url, record, base_dir, mode):
    """One pass for `mode` with budgeted thinking + forced-close fallback."""
    t0 = time.monotonic()
    messages = build_messages(record, base_dir, mode)
    ctx_overflow = False
    try:
        r1 = await chat(session, base_url, messages, THINK_BUDGET + ANSWER_BUDGET)
    except ContextOverflow:
        # Prompt + image tokens left too little room for the full think budget.
        # Retry with a much smaller budget so we still get an answer.
        ctx_overflow = True
        r1 = await chat(session, base_url, messages, 2048)

    reasoning, answer_text = r1["reasoning"], r1["content"]
    think_truncated = False

    # Spiral case: budget hit while still inside <think> (content empty,
    # finish_reason=length). Force the model to commit to an answer.
    if not answer_text.strip() and r1["finish_reason"] == "length":
        think_truncated = True
        # Follow-up turn: feed back the (truncated) thinking and demand only the
        # JSON answer. A plain new user turn is more robust across vLLM template
        # quirks than continue_final_message (which 400s on multi-image prompts).
        followup = messages + [
            {"role": "assistant",
             "content": f"<think>\n{reasoning}\n</think>"},
            {"role": "user",
             "content": 'Based on your reasoning above, give your final answer now '
                        'as ONLY this JSON object: {"answer": "X"} '
                        'where X is A, B, C, or D.'},
        ]
        try:
            r2 = await chat(session, base_url, followup, ANSWER_BUDGET)
            answer_text = r2["content"] or r2["reasoning"]
            r2usage = r2["usage"]
        except ContextOverflow:
            # No room even for the follow-up — fall back to parsing the
            # conclusion straight out of the truncated reasoning.
            answer_text = reasoning[-400:]
            r2usage = {}
        merged = {}
        for k in set(r1["usage"]) | set(r2usage):
            a, b = r1["usage"].get(k), r2usage.get(k)
            if isinstance(a, (int, float)) or isinstance(b, (int, float)):
                merged[k] = (a or 0) + (b or 0)
            else:
                merged[k] = a if a is not None else b
        r1["usage"] = merged

    letter = parse_letter(answer_text) or parse_letter(reasoning)
    return {
        "reasoning": reasoning,
        "answer_text": answer_text,
        "letter": letter,
        "finish_reason": r1["finish_reason"],
        "think_truncated": think_truncated,
        "ctx_overflow": ctx_overflow,
        "usage": r1["usage"],
        "latency_s": round(time.monotonic() - t0, 2),
    }


async def worker(name, queue, session, base_url, out_f, lock, counters):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        source, record, mode = item
        base_dir = SOURCES[source][1]
        gt = record.get("answer")
        try:
            res = await run_one(session, base_url, record, base_dir, mode)
            correct = res["letter"] == gt
            row = {
                "question_id": record["question_id"],
                "sample_id": record.get("sample_id"),
                "source": source, "mode": mode,
                "topic": record["topic"], "city": record.get("city"),
                "image_mode": record.get("image_mode"),
                "gt": gt, "letter": res["letter"], "correct": correct,
                "reasoning": res["reasoning"], "answer_text": res["answer_text"],
                "think_truncated": res["think_truncated"],
                "ctx_overflow": res["ctx_overflow"],
                "finish_reason": res["finish_reason"],
                "usage": res["usage"], "latency_s": res["latency_s"],
            }
            async with lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["done"] += 1
                counters[f"{mode}_correct"] += int(correct)
                counters[f"{mode}_n"] += 1
                if res["think_truncated"]:
                    counters["truncated"] += 1
        except Exception as e:  # one bad (record, mode) shouldn't kill the run
            async with lock:
                out_f.write(json.dumps({
                    "question_id": record["question_id"], "source": source,
                    "mode": mode, "error": repr(e),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["errors"] += 1
        finally:
            queue.task_done()
            if counters["done"] % 50 == 0:
                def pct(m):
                    n = counters[f"{m}_n"] or 1
                    return 100 * counters[f"{m}_correct"] / n
                print(f"  {counters['done']} passes | "
                      f"full {pct('full'):.0f}% sat {pct('sat_only'):.0f}% "
                      f"sv {pct('sv_only'):.0f}% blind {pct('blind'):.0f}% | "
                      f"trunc {counters['truncated']} err {counters['errors']}",
                      flush=True)


def load_done_keys(out_path):
    """Set of (source, question_id, mode) already logged — for resume."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "mode" in r and "question_id" in r and "error" not in r:
                done.add((r.get("source"), r["question_id"], r["mode"]))
    return done


async def main_async(args):
    # Build worklist: one (source, record, mode) item per applicable mode.
    work = []
    mode_counts = defaultdict(int)
    for source in args.sources:
        path, _ = SOURCES[source]
        recs = load_jsonl(path)
        if args.limit_per_topic:
            by_t = defaultdict(list)
            for r in recs:
                by_t[r["topic"]].append(r)
            recs = [r for rs in by_t.values() for r in rs[: args.limit_per_topic]]
        for r in recs:
            for mode in modes_for_topic(r["topic"]):
                work.append((source, r, mode))
                mode_counts[mode] += 1

    # Resume: skip (source, qid, mode) already logged
    done_keys = load_done_keys(args.out)
    if done_keys:
        before = len(work)
        work = [(s, r, m) for (s, r, m) in work
                if (s, r["question_id"], m) not in done_keys]
        print(f"Resume: {before - len(work)} (record,mode) pairs already done, "
              f"{len(work)} remain.", flush=True)

    print(f"Running {len(work)} generations across {args.sources} "
          f"@ concurrency {args.concurrency}", flush=True)
    print(f"  mode breakdown: {dict(mode_counts)}", flush=True)

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
    print(f"generations: {counters['done']} | truncated-think: {counters['truncated']} "
          f"| errors: {counters['errors']}")
    for m in ("full", "sat_only", "sv_only", "blind"):
        if counters[f"{m}_n"]:
            print(f"  {m:9s}: {counters[f'{m}_correct']}/{counters[f'{m}_n']} correct")
    print(f"log: {args.out}")
    print("Run analyze.py for the per-topic per-mode table.")


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
