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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary as summ  # noqa: E402  (live structured summary builder)

ROOT = Path("/home/ezel/Development/EOLLM/dataset_content/EODATA_compressed_final")
RESULTS_DIR = Path("/home/ezel/Development/EOLLM/reasoning_distill/results")

# Set from CLI in main(); stamped into every row so a combined log stays
# attributable per model.
MODEL_LABEL = "unknown"
# The name the vLLM server answers to (its --served-model-name). Different
# servers may use different names; settable so we can target any server.
SERVED_NAME = "teacher"

# Live-summary cadence: refresh at least every this many completed generations,
# AND at least every this many seconds — whichever comes first — so the live
# view never goes stale and the final partial batch (<100) is never missed.
SUMMARY_EVERY_N = 100
SUMMARY_EVERY_S = 60

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

THINK_BUDGET = 8000    # tokens allowed inside <think> before we force-close.
                       # 8k lets most traces complete naturally (good distillation
                       # material); spilled-over thinking is force-closed into a
                       # JSON answer so an answer is always captured.
ANSWER_BUDGET = 256    # tokens for the forced final answer in phase 2 — roomy
                       # enough that the model can restate before the JSON.
PARSE_RETRIES = 4      # if no A/B/C/D parses out, re-run the pass up to 4x
                       # (a malformed/missing JSON is a parse miss, not a real
                       # answer — give it patience before recording a failure).
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

    if mode == "satellite_only":
        sat = [res["primary"]]                       # 1 raw sat, no SV exists

    elif mode == "satellite_marked":
        # Urban-attribute questions (land_use, building_height, urban_density,
        # junction_type, green_space, amenity_richness, road_type, road_surface,
        # transit_density). DESIGNED dual-perspective: 1 marked sat + 4 SV angles.
        # Mirror training/data.py's satellite_marked branch EXACTLY — the SV
        # images are labeled by viewing DIRECTION (Fwd/Bwd/Left/Right), NOT
        # A/B/C/D, because here the A/B/C/D letters are the answer choices, so a
        # letter label on an image would collide with the answer.
        imgs = record["images"]
        sat = [cu.make_sat_marked(os.path.join(base, imgs["satellite"]))]
        angle_label = {
            "along_fwd": "Fwd", "along_bwd": "Bwd",
            "cross_left": "Left", "cross_right": "Right",
        }
        for angle in cu.STV_ANGLES:
            sv_rel = imgs.get(f"streetview_{angle}")
            if not sv_rel:
                continue  # rare missing angle (London SV gaps) — skip, don't fail
            sv.append(add_corner_label(cu.Image.open(os.path.join(base, sv_rel)),
                                       angle_label[angle]))

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


# SCOPE: this ablation covers ONLY the 9 attribute-classification tasks. They are
# DESIGNED dual-perspective (marked sat + 4 SV angles, per the fixed training
# loader), so full=both, sat_only=marked sat, sv_only=4 SV, blind=none are all
# meaningful — exactly the "does the model fuse the two views?" question.
# (Before the loader fix these silently fed satellite-only — the 62.9%-of-dataset
# bug; see project_satellite_marked_sv_bug memory.)
BOTH_PERSPECTIVE_FULL = {  # full / sat_only / sv_only / blind
    "land_use", "building_height", "urban_density", "junction_type",
    "green_space", "amenity_richness", "road_type", "road_surface",
    "transit_density",
}

# Cross-view MATCHING tasks — excluded from this ablation entirely. They are not
# attribute classification: the task itself IS aligning two perspectives, so a
# single-perspective ablation is ill-posed. (mismatch_* needs both images to even
# define the question; camera_direction's sat_only drops the query SV and sv_only
# drops the arrow-satellites that ARE the options.) Kept here so the worklist
# skips them; their existing rows stay on disk but are no longer generated.
EXCLUDED_CROSSVIEW = {
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard",
    "camera_direction",
}


def modes_for_topic(topic):
    if topic in EXCLUDED_CROSSVIEW:
        return []  # cross-view matching — out of scope for this ablation
    if topic in BOTH_PERSPECTIVE_FULL:
        return ["full", "sat_only", "sv_only", "blind"]
    return ["full", "blind"]  # any other satellite_marked-style topic


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
        "model": SERVED_NAME, "messages": messages,
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


async def run_one_with_retries(session, base_url, record, base_dir, mode):
    """run_one, but if no A/B/C/D parses out, retry up to PARSE_RETRIES times.
    A missing/malformed answer is a parse miss (model rambled, bad JSON), not a
    genuine response — give it patience before recording a failed parse."""
    res = None
    for attempt in range(PARSE_RETRIES + 1):
        res = await run_one(session, base_url, record, base_dir, mode)
        if res["letter"]:
            res["parse_attempts"] = attempt + 1
            return res
    # exhausted retries — return the last attempt with letter=None
    res["parse_attempts"] = PARSE_RETRIES + 1
    return res


async def worker(name, queue, session, base_url, out_f, lock, counters,
                 all_rows, maybe_flush_summary_locked=None):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        source, record, mode = item
        base_dir = SOURCES[source][1]
        gt = record.get("answer")
        try:
            res = await run_one_with_retries(session, base_url, record, base_dir, mode)
            correct = res["letter"] == gt
            row = {
                "model": MODEL_LABEL,
                "question_id": record["question_id"],
                "sample_id": record.get("sample_id"),
                "source": source, "mode": mode,
                "topic": record["topic"], "city": record.get("city"),
                "image_mode": record.get("image_mode"),
                "gt": gt, "letter": res["letter"], "correct": correct,
                "reasoning": res["reasoning"], "answer_text": res["answer_text"],
                "think_truncated": res["think_truncated"],
                "ctx_overflow": res["ctx_overflow"],
                "parse_attempts": res.get("parse_attempts", 1),
                "finish_reason": res["finish_reason"],
                "usage": res["usage"], "latency_s": res["latency_s"],
            }
            async with lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                all_rows.append(row)          # in-memory mirror for live summary
                counters["done"] += 1
                counters[f"{mode}_correct"] += int(correct)
                counters[f"{mode}_n"] += 1
                if res["think_truncated"]:
                    counters["truncated"] += 1
                self_log(counters)
                if maybe_flush_summary_locked is not None:
                    maybe_flush_summary_locked()  # sync, under lock
        except Exception as e:  # one bad (record, mode) shouldn't kill the run
            async with lock:
                out_f.write(json.dumps({
                    "model": MODEL_LABEL,
                    "question_id": record["question_id"], "source": source,
                    "mode": mode, "error": repr(e),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["errors"] += 1
        finally:
            queue.task_done()


def self_log(counters):
    """Periodic stdout progress line (called under lock)."""
    if counters["done"] % 50 == 0 and counters["done"] > 0:
        def pct(m):
            n = counters[f"{m}_n"] or 1
            return 100 * counters[f"{m}_correct"] / n
        print(f"  {counters['done']} done | "
              f"full {pct('full'):.0f}% sat {pct('sat_only'):.0f}% "
              f"sv {pct('sv_only'):.0f}% blind {pct('blind'):.0f}% | "
              f"trunc {counters['truncated']} err {counters['errors']}",
              flush=True)


def write_summary(out_dir, rows, meta):
    """Build the structured summary from the IN-MEMORY rows and write
    summary.json + .md. Avoids re-parsing the (growing) JSONL each flush —
    critical at 100k+ rows where a re-read would block the event loop."""
    s = summ.build_summary(rows, meta=meta)
    # atomic-ish write (write temp then replace) so a reader never sees a partial file
    for name, payload in (("summary.json", json.dumps(s, indent=2, ensure_ascii=False)),
                          ("summary.md", summ.render_markdown(s))):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(payload)
        tmp.replace(out_dir / name)
    return s


async def summary_refresher(out_dir, all_rows, meta, counters, stop_evt, total, t_start):
    """Background task: refresh the live summary every SUMMARY_EVERY_S seconds
    from the in-memory rows. Guarantees a time-based floor between N-triggers."""
    while not stop_evt.is_set():
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=SUMMARY_EVERY_S)
        except asyncio.TimeoutError:
            pass
        done = counters["done"]
        elapsed = max(1e-6, time.monotonic() - t_start)
        gpm = round(done / elapsed * 60, 1)
        remaining = max(0, total - done)
        eta_min = round(remaining / gpm, 1) if gpm > 0 else None
        meta.update({
            "updated": _now_iso(),
            "gens_per_min": gpm,
            "eta": f"{eta_min} min" if eta_min is not None else None,
        })
        write_summary(out_dir, list(all_rows), meta)


def _now_iso():
    # Date.now() is unavailable in some sandboxes; use time.time via strftime.
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main_async(args):
    global MODEL_LABEL, SERVED_NAME
    MODEL_LABEL = args.label
    SERVED_NAME = args.served_name

    out_dir = RESULTS_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "ablation_log.jsonl")

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
                if args.modes and mode not in args.modes:
                    continue  # --modes restricts which passes run this invocation
                work.append((source, r, mode))
                mode_counts[mode] += 1
    total_planned = len(work)

    # Resume: skip (source, qid, mode) already logged in THIS model's log, and
    # PRELOAD the prior rows so the live summary reflects total progress (not
    # just this process's new work) and we never re-parse the file mid-run.
    all_rows = summ.load_rows(log_path) if os.path.exists(log_path) else []
    all_rows = [r for r in all_rows if "error" not in r]  # drop old error rows
    done_keys = {(r.get("source"), r.get("question_id"), r.get("mode"))
                 for r in all_rows if r.get("question_id") and r.get("mode")}
    if done_keys:
        before = len(work)
        work = [(s, r, m) for (s, r, m) in work
                if (s, r["question_id"], m) not in done_keys]
        print(f"Resume: {before - len(work)} (record,mode) pairs already done, "
              f"{len(work)} remain.", flush=True)

    print(f"[{args.label}] Running {len(work)} generations across {args.sources} "
          f"@ concurrency {args.concurrency}", flush=True)
    print(f"  mode breakdown: {dict(mode_counts)}", flush=True)
    print(f"  results dir: {out_dir}", flush=True)

    meta = {
        "model": args.label,
        "model_name": args.model_name,
        "served_name": args.served_name,
        "base_url": args.base_url,
        "sources": args.sources,
        "think_budget": THINK_BUDGET,
        "answer_budget": ANSWER_BUDGET,
        "parse_retries": PARSE_RETRIES,
        "temperature": TEMPERATURE,
        "concurrency": args.concurrency,
        "total": total_planned,
        "already_done": total_planned - len(work),
        "started": _now_iso(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    counters = defaultdict(int)
    counters["_summary_dirty_at"] = 0
    queue = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    out_f = open(log_path, "a")
    lock = asyncio.Lock()
    t_start = time.monotonic()
    stop_evt = asyncio.Event()
    done_at_start = len(all_rows)  # for accurate throughput on resume
    # N-based summary flush — MUST be called while holding `lock` (it reads the
    # shared counters/all_rows). The dirty_at guard makes it fire once per N
    # even with many workers contending.
    def maybe_flush_summary_locked():
        if counters["done"] - counters["_summary_dirty_at"] >= SUMMARY_EVERY_N:
            counters["_summary_dirty_at"] = counters["done"]
            elapsed = max(1e-6, time.monotonic() - t_start)
            new_done = counters["done"] - done_at_start
            gpm = round(new_done / elapsed * 60, 1)
            meta.update({"updated": _now_iso(), "gens_per_min": gpm,
                         "eta": f"{round(max(0,total_planned-counters['done'])/gpm,1)} min" if gpm else None})
            write_summary(out_dir, list(all_rows), meta)

    timeout = aiohttp.ClientTimeout(total=1200)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        refresher = asyncio.create_task(
            summary_refresher(out_dir, all_rows, meta, counters, stop_evt,
                              total_planned, t_start))
        workers = [
            asyncio.create_task(
                worker(f"w{i}", queue, session, args.base_url, out_f, lock,
                       counters, all_rows, maybe_flush_summary_locked))
            for i in range(args.concurrency)
        ]
        await queue.join()
        await asyncio.gather(*workers)
        stop_evt.set()
        await refresher
    out_f.close()

    # FINAL flush — guarantees the last partial batch (<SUMMARY_EVERY_N) is
    # captured. Rebuild from the LOG (authoritative, includes error rows) so the
    # final summary is complete even if all_rows missed anything.
    meta["updated"] = _now_iso()
    meta["finished"] = _now_iso()
    final = write_summary(out_dir, summ.load_rows(log_path), meta)

    print("\n=== DONE ===", flush=True)
    print(f"generations: {counters['done']} | truncated-think: {counters['truncated']} "
          f"| errors: {counters['errors']}")
    print(f"overall acc: {final['progress']['overall_acc']}%")
    for m in ("full", "sat_only", "sv_only", "blind"):
        if counters[f"{m}_n"]:
            print(f"  {m:9s}: {counters[f'{m}_correct']}/{counters[f'{m}_n']} correct")
    print(f"results: {out_dir}/  (ablation_log.jsonl, summary.json, summary.md)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--served-name", default="teacher",
                    help="the name the vLLM server answers to (--served-model-name)")
    ap.add_argument("--model-name", default="qwen3.5-27b",
                    help="human model identifier; label defaults to <model-name>_<timestamp>")
    ap.add_argument("--served-model", dest="model_name",
                    help="(alias of --model-name)")
    ap.add_argument("--label", default=None,
                    help="run label = results/<label>/ dir. Default: <model-name>_<timestamp>")
    ap.add_argument("--sources", nargs="+",
                    default=["train", "validation", "benchmark"],
                    choices=list(SOURCES))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--modes", nargs="+", default=None,
                    choices=["full", "sat_only", "sv_only", "blind"],
                    help="restrict to these passes (default: all applicable per "
                         "topic). E.g. --modes sat_only sv_only full")
    ap.add_argument("--limit-per-topic", type=int, default=0,
                    help="cap records per topic per source (smoke testing)")
    args = ap.parse_args()
    if not args.label:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = args.model_name.replace("/", "-")
        args.label = f"{safe}_{ts}"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
