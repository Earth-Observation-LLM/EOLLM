#!/usr/bin/env python3
"""Prompting-STRATEGY ablation against a vLLM-served VLM (remote lab-ws).

QUESTION
========
Our perspective ablation showed that on the 9 urban-attribute tasks the model
gets ~nothing from the second view (full ≈ sat_only ≈ sv_only). Can we change
HOW we prompt — not the model — to make it actually fuse both views / boost
accuracy? We compare THREE prompting strategies head-to-head on the SAME model,
SAME records, SAME four view-ablations, so any delta is attributable to the
strategy alone.

STRATEGIES (--strategy)
=======================
  current    Thinking OFF, single turn: ask directly with the 4 options, demand
             a strict JSON answer. The baseline (= what we run today).
  think16k   Thinking ON, single turn, <think> budget 16k tokens, force-close
             fallback if it spirals past budget (so an answer is always parsed).
  multistep  Thinking OFF, a 4-turn GUIDED conversation:
               T1 observe   — describe what each image shows (per-view checklist),
                              no answering yet.
               T2 reason    — answer the question in words WITHOUT the options.
               T3 self-check— "are you sure? did you overlook / assume anything?"
               T4 commit    — now shown the options, output ONLY {"answer":"X"}.
             We SCORE turn 4.

VIEWS (--modes), per record, all four by default:
  full      marked-satellite + 4-angle street-view grid
  sat_only  marked-satellite only
  sv_only   street-view grid only
  blind     no images (text + options only) — control for leakage

SCOPE: the 9 attribute-classification tasks only (image_mode satellite_marked).
Cross-view matching tasks (mismatch_*, camera_direction) are excluded — a single-
view ablation is ill-posed there.

IMAGES come pre-rendered on lab-ws under EarthMLLMEval/data/images/, addressed by
the record's own fields (images.satellite_marked, images.stv_composite_labeled),
so they are byte-identical to what the benchmark harness feeds. NO local dataset
or composite_utils dependency.

LOGGING is per (question_id, mode, strategy): the parsed letter, correctness, the
FULL multi-turn transcript (every turn's prompt + the model's reasoning/answer),
token usage, finish reason, latency, whether the think budget was hit, and parse
retries. One JSONL line per (question_id, mode). RESUMABLE: already-logged
(source, question_id, mode, strategy) keys are skipped on restart.

vLLM must be served with --reasoning-parser qwen3 so <think> comes back split into
`reasoning` and the answer into `content`. Thinking is toggled PER REQUEST via the
chat-template kwarg enable_thinking (passed as a top-level `chat_template_kwargs`
field in the request body).

No frontier API is touched — local vLLM only.

Usage (server up):
  python run_strategy_ablation.py --strategy current   --modes full sat_only sv_only blind
  python run_strategy_ablation.py --strategy multistep --limit-per-topic 5   # smoke
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summary as summ  # noqa: E402  (live structured summary builder, copied verbatim)

# ---------------------------------------------------------------------------
# Paths. DATA_ROOT is the EarthMLLMEval data dir on lab-ws; image paths in the
# benchmark JSONL are relative to it ("images/sat_marked/...", "images/...").
# Overridable via CLI for portability.
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path("/home/ain480/evaluation/EarthMLLMEval/data")
DEFAULT_BENCHMARK = DEFAULT_DATA_ROOT / "benchmark_with_answers.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Stamped into every row so a combined log stays attributable.
MODEL_LABEL = "qwen9b"
SERVED_NAME = "qwen9b"
STRATEGY = "current"

SUMMARY_EVERY_N = 100
SUMMARY_EVERY_S = 60

# The 9 attribute-classification tasks — DESIGNED dual-perspective (marked sat +
# 4-angle SV grid). These are the ONLY topics in scope.
ATTR_TOPICS = {
    "land_use", "building_height", "urban_density", "junction_type",
    "green_space", "amenity_richness", "road_type", "road_surface",
    "transit_density",
}
# Out of scope — single-view ablation is ill-posed (the task IS aligning views).
EXCLUDED_TOPICS = {
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard", "camera_direction",
}

MODES = ["full", "sat_only", "sv_only", "blind"]

# ---- generation knobs ------------------------------------------------------
# greedy / temp 0 everywhere = full determinism & reproducible transcripts.
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1

THINK_BUDGET = 16000     # think16k: tokens allowed inside <think> before force-close.
ANSWER_BUDGET = 256      # tokens for a forced/normal final JSON answer.

# MULTISTEP: do NOT cap the conversational turns. The smoke run showed 33% of
# observe turns truncating at 600 tok and producing cut-off descriptions, which
# weakens the downstream reasoning and would unfairly handicap multistep in the
# head-to-head. Per the design intent, the model writes as much as it wants on
# observe / reason / self_check — bounded only by the model's context window.
# We pass a large ceiling (effectively "no cap": vLLM clamps it to whatever
# context remains after the prompt). The 24576-token context + a small prompt
# leaves >20k tokens of room, far beyond any natural description.
TURN_BUDGET_UNCAPPED = 20000  # ~= no cap; vLLM clamps to remaining context.
OBSERVE_BUDGET = TURN_BUDGET_UNCAPPED   # multistep T1: image description
REASON_BUDGET = TURN_BUDGET_UNCAPPED    # multistep T2: free-form reasoning
CHECK_BUDGET = TURN_BUDGET_UNCAPPED     # multistep T3: self-check
# The COMMIT turn is the ONLY one that should be short — it must emit just the
# JSON. Kept roomy enough that a preface + the JSON never truncates mid-object
# (the M1 fix), but not unbounded since it's meant to be terminal.
COMMIT_BUDGET = 256      # multistep T4 / current: the final JSON answer
PARSE_RETRIES = 4        # re-run a pass up to N times if no A/B/C/D parses out


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def pil_to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ===========================================================================
# IMAGES — remote, pre-rendered. Mirror EarthMLLMEval/eval/utils.py's normal-
# question branch: marked satellite + 4-angle labeled street-view composite.
# ===========================================================================
from PIL import Image  # noqa: E402


def _abs(data_root, rel):
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else Path(data_root) / rel


def view_images(record, data_root, mode):
    """Return the PIL image list for a given view of an attribute record.

    full     -> [marked_sat, stv_composite_labeled]
    sat_only -> [marked_sat]
    sv_only  -> [stv_composite_labeled]
    blind    -> []

    Missing files are skipped (rare SV gaps), never fatal. Built straight from
    the record's own image fields so it matches the benchmark harness exactly.
    """
    if mode == "blind":
        return []
    img = record.get("images", {}) or {}
    sat_rel = img.get("satellite_marked") or record.get("sat_marked_path")
    sv_rel = img.get("stv_composite_labeled") or record.get("composite_stv_labeled_path")

    want_sat = mode in ("full", "sat_only")
    want_sv = mode in ("full", "sv_only")
    out = []
    if want_sat:
        p = _abs(data_root, sat_rel)
        if p and p.exists():
            out.append(Image.open(p))
    if want_sv:
        p = _abs(data_root, sv_rel)
        if p and p.exists():
            out.append(Image.open(p))
    return out


def expected_image_count(mode):
    """How many images a view MUST carry. full=2 (sat+sv), sat_only/sv_only=1,
    blind=0. Used as a send-time tripwire (H1)."""
    return {"full": 2, "sat_only": 1, "sv_only": 1, "blind": 0}[mode]


def has_required_images(record, data_root, mode):
    """True iff the view can be built with ALL its expected images (used to skip
    records with missing files; blind is always buildable). Requires the FULL
    expected count, not just >0, so `full` with a missing SV is skipped rather
    than silently degraded to sat_only."""
    if mode == "blind":
        return True
    return len(view_images(record, data_root, mode)) == expected_image_count(mode)


def attach_images(content, record, data_root, mode):
    """Append the view's images to a chat `content` list, with a HARD tripwire
    (H1): a non-blind view that resolves to the wrong image count RAISES instead
    of silently sending a blind/degraded prompt that would still be scored as a
    valid `full`/`sat_only`/`sv_only` answer — which would corrupt the very
    'does the second view help?' comparison this experiment exists to make.
    The raise is caught by the worker and logged as an `error` row."""
    imgs = view_images(record, data_root, mode)
    want = expected_image_count(mode)
    if len(imgs) != want:
        raise RuntimeError(
            f"image count mismatch for qid={record.get('question_id')} "
            f"mode={mode}: got {len(imgs)} expected {want}")
    for img in imgs:
        content.append({"type": "image_url", "image_url": {"url": pil_to_b64(img)}})
    return content


def options_block(record):
    return "\n".join(f"{k}. {v}" for k, v in record["options"].items())


# ===========================================================================
# PROMPTS — the experiment. Written deliberately, task-aware.
#
# Task: 4-way multiple-choice on an URBAN ATTRIBUTE of one location. Inputs:
#   - a SATELLITE image with the location of interest MARKED (ring/dot), and
#   - a 2x2 GRID of 4 STREET-VIEW angles along the road at that location
#     (forward / backward / left / right), corner-labeled.
# Every prompt stays HONEST about which of these is actually shown for the view,
# so the model never claims a perspective it cannot see.
# ===========================================================================

# What the model is, per view — honest about the inputs shown.
_ROLE = {
    "full":     "an urban-geography expert analysing a marked satellite image together with street-level photography",
    "sat_only": "an urban-geography expert analysing a marked satellite image",
    "sv_only":  "an urban-geography expert analysing street-level photography",
    "blind":    "an urban-geography expert",
}

# A neutral description of the imagery shown, reused across strategies.
_IMG_NOTE = {
    "full":     ("You are shown TWO things: (1) a satellite image in which the "
                 "location of interest is MARKED, and (2) a grid of FOUR "
                 "street-level photos taken at that location looking forward, "
                 "backward, left and right along the road."),
    "sat_only": ("You are shown a satellite image in which the location of "
                 "interest is MARKED."),
    "sv_only":  ("You are shown a grid of FOUR street-level photos taken at one "
                 "location looking forward, backward, left and right along the "
                 "road."),
    "blind":    ("No imagery is available for this question — answer from the "
                 "question text and options alone."),
}

_JSON_RULE = ('Output ONLY a JSON object on its own line and nothing else: '
              '{"answer": "X"} where X is exactly one of A, B, C, or D.')
# Mid-sentence variant (lowercase lead-in) for when the rule follows other text.
_JSON_RULE_MID = ('output ONLY a JSON object on its own line and nothing else: '
                  '{"answer": "X"} where X is exactly one of A, B, C, or D.')


# ---- per-view observation checklist (multistep T1) ------------------------
_OBS_SAT = (
    "From the SATELLITE image (focus on the marked location and its immediate "
    "surroundings), describe concretely: the road layout and grid pattern; "
    "building footprints — their size, spacing and density; any green space, "
    "parks, water or open ground; large flat roofs, parking lots or industrial "
    "structures; and the overall built character (dense urban / suburban / "
    "open / industrial)."
)
_OBS_SV = (
    "From the FOUR STREET-LEVEL photos, describe concretely what is visible "
    "across the angles: building heights (in storeys) and how built-up it is; "
    "shopfronts, signage and commercial activity vs purely residential; the "
    "road surface and width, lane markings, sidewalks; vegetation and street "
    "trees; and any transit infrastructure (rails, stops, wires)."
)


def sys_prompt(mode, strategy):
    base = f"You are {_ROLE[mode]}, answering a multiple-choice question about one location."
    if strategy == "think16k":
        # Honest about whether there's imagery to reason over (blind has none).
        think = (" Think step by step about what the images actually show before "
                 "deciding." if mode != "blind"
                 else " Think step by step before deciding.")
        return (base + think + " Reason carefully but do not pad — reach a "
                "well-supported decision. After your reasoning, " + _JSON_RULE_MID)
    if strategy == "current":
        return base + " " + _JSON_RULE
    # multistep — the per-turn user prompts carry the instructions; keep the
    # system role honest and minimal, and explicitly disable rambling.
    return base + " Follow the conversation step by step. Be concrete and concise."


def single_turn_messages(record, data_root, mode, strategy):
    """Build the one-shot messages for `current` and `think16k`."""
    text = (f"{_IMG_NOTE[mode]}\n\n"
            f"Question: {record['question']}\n\n"
            f"Options:\n{options_block(record)}\n\n"
            f"{_JSON_RULE}")
    content = [{"type": "text", "text": text}]
    attach_images(content, record, data_root, mode)  # H1 tripwire
    return [
        {"role": "system", "content": sys_prompt(mode, strategy)},
        {"role": "user", "content": content},
    ]


def multistep_turn1(record, data_root, mode):
    """T1 user message — observe. Per-view tailored checklist. Images attach HERE
    only. For blind there is no image step, so T1 is skipped by the caller."""
    if mode == "full":
        instr = (_IMG_NOTE["full"] + "\n\nLook carefully. Do NOT answer any "
                 "question yet — just describe what you observe.\n\n"
                 "(a) " + _OBS_SAT + "\n\n(b) " + _OBS_SV + "\n\n"
                 "Finally, note any point where the satellite and the street-level "
                 "views AGREE or DISAGREE about the character of the place.")
    elif mode == "sat_only":
        instr = (_IMG_NOTE["sat_only"] + "\n\nLook carefully. Do NOT answer any "
                 "question yet — just describe what you observe.\n\n" + _OBS_SAT)
    elif mode == "sv_only":
        instr = (_IMG_NOTE["sv_only"] + "\n\nLook carefully. Do NOT answer any "
                 "question yet — just describe what you observe.\n\n" + _OBS_SV)
    else:
        raise ValueError(f"multistep_turn1 not used for mode={mode}")
    content = [{"type": "text", "text": instr}]
    attach_images(content, record, data_root, mode)  # H1 tripwire
    return {"role": "user", "content": content}


def multistep_turn2(record, mode):
    """T2 — reason WITHOUT options."""
    if mode == "blind":
        lead = ("Consider the following question about one location. You have no "
                "imagery, so reason from general urban-geography knowledge.")
    else:
        lead = ("Based ONLY on what you described above, answer the following "
                "question in your own words and explain your reasoning.")
    return {"role": "user", "content": (
        f"{lead}\n\nQuestion: {record['question']}\n\n"
        "Do NOT pick a letter — there are no options shown yet. Give your "
        "best answer in a sentence or two, with the reasoning that supports it."
    )}


def multistep_turn3():
    """T3 — self-check."""
    return {"role": "user", "content": (
        "Now re-read your own reasoning critically. Are you confident? Did you "
        "overlook something in the image(s), or make an assumption the evidence "
        "does not support? If you find a mistake, correct it now; otherwise "
        "restate your conclusion concisely."
    )}


def multistep_turn4(record):
    """T4 — commit. Options revealed; scored."""
    return {"role": "user", "content": (
        f"Here are the answer options:\n{options_block(record)}\n\n"
        "Choose the SINGLE option most consistent with your reasoning above. "
        + _JSON_RULE
    )}


# ===========================================================================
# Letter parsing — ONLY trust an explicit commitment, never scrape prose.
#
# ZERO-TRUST review (C2): the old last-standalone-A/B/C/D fallback fabricated
# letters from echoed option labels ("A. residential B. commercial …" -> B) and
# from rejected mid-reasoning letters ("between C and D, go with D"). Both
# systematically mis-score. We therefore parse ONLY:
#   1) the {"answer":"X"} JSON we explicitly ask for (last one wins),
#   2) a reply that is JUST a letter ("C", "(C).", "C:"),
#   3) an explicit "answer is X" / "answer: X" near the end.
# Anything else -> None (an honest unparsed miss, which the caller retries).
# The bare-letter catch-all is GONE — under temp-0 truncation a guessed letter
# is worse than a miss, because it biases the head-to-head accuracy comparison.
# ===========================================================================
def parse_letter(text):
    import re
    if not text:
        return None
    t = text.strip()
    # 1) JSON answer object (what every prompt requests) — take the LAST one.
    js = re.findall(r'"answer"\s*:\s*"?\(?([ABCD])\)?"?', t, re.I)
    if js:
        return js[-1].upper()
    # 2) the whole reply is just a letter.
    m = re.fullmatch(r"\(?([ABCD])\)?[.):]?", t, re.I)
    if m:
        return m.group(1).upper()
    # 3) explicit "answer is X" / "answer: X" — only when there is exactly ONE
    #    such phrase, so we never pick between competing commitments.
    ans = re.findall(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", t, re.I)
    if len(ans) == 1:
        return ans[0].upper()
    if len(ans) > 1 and len(set(a.upper() for a in ans)) == 1:
        return ans[0].upper()  # repeated but consistent
    # No unambiguous commitment -> honest miss (caller retries).
    return None


class ContextOverflow(Exception):
    """A 400 from the server, almost always max-context exceeded."""


async def chat(session, base_url, messages, max_tokens, _attempt=0, **extra):
    """One /v1/chat/completions call. `extra` is merged into the request body —
    that's how thinking is toggled: chat(..., chat_template_kwargs={'enable_thinking': False})."""
    payload = {
        "model": SERVED_NAME, "messages": messages,
        "max_tokens": max_tokens, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K, **extra,
    }
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as r:
            if r.status == 400:
                raise ContextOverflow(await r.text())
            r.raise_for_status()
            data = await r.json()
    except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError):
        if _attempt >= 4:
            raise
        await asyncio.sleep(2 ** _attempt)
        return await chat(session, base_url, messages, max_tokens,
                          _attempt=_attempt + 1, **extra)
    choice = data["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "reasoning": reasoning,
        "content": msg.get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}) or {},
    }


THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _merge_usage(a, b):
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) or isinstance(y, (int, float)):
            out[k] = (x or 0) + (y or 0)
        else:
            out[k] = x if x is not None else y
    return out


# ===========================================================================
# Strategy runners. Each returns a normalised result dict.
# ===========================================================================
async def run_think16k(session, base_url, record, data_root, mode):
    """Single turn, thinking ON, budgeted with force-close fallback."""
    messages = single_turn_messages(record, data_root, mode, "think16k")
    ctx_overflow = False
    transcript = [{"turn": "ask", "role": "user",
                   "text": messages[-1]["content"][0]["text"]}]
    try:
        r1 = await chat(session, base_url, messages, THINK_BUDGET + ANSWER_BUDGET)
    except ContextOverflow:
        ctx_overflow = True
        r1 = await chat(session, base_url, messages, 2048)
    reasoning, answer_text, usage = r1["reasoning"], r1["content"], r1["usage"]
    think_truncated = False

    # Spiral: budget hit while still inside <think> (no content, finish=length).
    if not answer_text.strip() and r1["finish_reason"] == "length":
        think_truncated = True
        followup = messages + [
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>"},
            {"role": "user", "content":
                'Based on your reasoning above, give your final answer now as '
                'ONLY this JSON object: {"answer": "X"} where X is A, B, C, or D.'},
        ]
        try:
            r2 = await chat(session, base_url, followup, ANSWER_BUDGET)
            # Score the FORCE-CLOSE turn's content only — never scrape the
            # reasoning trace (C1): a letter from rejected mid-thinking
            # ("between C and D… go with D") would mis-score.
            answer_text = r2["content"]
            usage = _merge_usage(usage, r2["usage"])
        except ContextOverflow:
            # No room even for the follow-up → genuinely unrecoverable.
            # Leave answer_text empty: an honest unparsed miss, which
            # run_one_with_retries will retry rather than a guessed letter.
            answer_text = ""
    transcript.append({"turn": "answer", "role": "assistant",
                       "reasoning": reasoning, "content": answer_text})
    # C1: parse ONLY the final answer text, NOT the 16k reasoning body.
    letter = parse_letter(answer_text)
    return {
        "letter": letter, "answer_text": answer_text, "reasoning": reasoning,
        "transcript": transcript, "finish_reason": r1["finish_reason"],
        "think_truncated": think_truncated, "ctx_overflow": ctx_overflow,
        "usage": usage,
    }


async def run_current(session, base_url, record, data_root, mode):
    """Single turn, thinking OFF, direct JSON answer."""
    messages = single_turn_messages(record, data_root, mode, "current")
    transcript = [{"turn": "ask", "role": "user",
                   "text": messages[-1]["content"][0]["text"]}]
    ctx_overflow = False
    try:
        r = await chat(session, base_url, messages, COMMIT_BUDGET, **THINK_OFF)
    except ContextOverflow:
        ctx_overflow = True
        r = await chat(session, base_url, messages, COMMIT_BUDGET)
    # H2: thinking is OFF, so score the answer CONTENT only — never reasoning.
    answer_text = r["content"]
    transcript.append({"turn": "answer", "role": "assistant",
                       "reasoning": r["reasoning"], "content": r["content"]})
    letter = parse_letter(answer_text)
    return {
        "letter": letter, "answer_text": answer_text, "reasoning": r["reasoning"],
        "transcript": transcript, "finish_reason": r["finish_reason"],
        "think_truncated": False, "ctx_overflow": ctx_overflow, "usage": r["usage"],
    }


async def run_multistep(session, base_url, record, data_root, mode):
    """4-turn guided conversation, thinking OFF, images on T1 only. Score T4."""
    msgs = [{"role": "system", "content": sys_prompt(mode, "multistep")}]
    transcript = []
    usage = {}
    ctx_overflow = False

    async def step(user_msg, budget, tag):
        nonlocal usage, ctx_overflow
        msgs.append(user_msg)
        # Log the user turn's text (drop image blobs from the transcript).
        if isinstance(user_msg["content"], list):
            utext = next((c["text"] for c in user_msg["content"]
                          if c.get("type") == "text"), "")
            n_imgs = sum(1 for c in user_msg["content"]
                         if c.get("type") == "image_url")
        else:
            utext, n_imgs = user_msg["content"], 0
        try:
            r = await chat(session, base_url, msgs, budget, **THINK_OFF)
        except ContextOverflow:
            ctx_overflow = True
            r = await chat(session, base_url, msgs, budget)
        reply = r["content"] or r["reasoning"]
        msgs.append({"role": "assistant", "content": reply})
        usage = _merge_usage(usage, r["usage"])
        transcript.append({"turn": tag, "n_images": n_imgs, "prompt": utext,
                           "reasoning": r["reasoning"], "content": reply,
                           "finish_reason": r["finish_reason"]})
        return r

    # T1 observe — skipped for blind (no imagery).
    if mode != "blind":
        await step(multistep_turn1(record, data_root, mode), OBSERVE_BUDGET, "observe")
    # T2 reason without options.
    await step(multistep_turn2(record, mode), REASON_BUDGET, "reason")
    # T3 self-check.
    await step(multistep_turn3(), CHECK_BUDGET, "self_check")
    # T4 commit — SCORED.
    r4 = await step(multistep_turn4(record), COMMIT_BUDGET, "commit")

    # H2: thinking OFF — score the commit turn's CONTENT only.
    answer_text = r4["content"]
    letter = parse_letter(answer_text)
    return {
        "letter": letter, "answer_text": answer_text, "reasoning": r4["reasoning"],
        "transcript": transcript, "finish_reason": r4["finish_reason"],
        "think_truncated": False, "ctx_overflow": ctx_overflow, "usage": usage,
    }


STRATEGY_RUNNERS = {
    "current": run_current,
    "think16k": run_think16k,
    "multistep": run_multistep,
}


async def run_one(session, base_url, record, data_root, mode, strategy):
    t0 = time.monotonic()
    res = await STRATEGY_RUNNERS[strategy](session, base_url, record, data_root, mode)
    res["latency_s"] = round(time.monotonic() - t0, 2)
    return res


async def run_one_with_retries(session, base_url, record, data_root, mode, strategy):
    """run_one, retried up to PARSE_RETRIES if no A/B/C/D parses (greedy, so a
    miss means a malformed/missing JSON, not a genuine answer)."""
    res = None
    for attempt in range(PARSE_RETRIES + 1):
        res = await run_one(session, base_url, record, data_root, mode, strategy)
        if res["letter"]:
            res["parse_attempts"] = attempt + 1
            return res
    res["parse_attempts"] = PARSE_RETRIES + 1
    return res


# ===========================================================================
# Worker / queue / live-summary scaffolding (mirrors run_ablation.py).
# ===========================================================================
async def worker(name, queue, session, base_url, data_root, out_f, lock,
                 counters, all_rows, maybe_flush_summary_locked=None):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        source, record, mode = item
        gt = record.get("answer")
        try:
            res = await run_one_with_retries(session, base_url, record, data_root,
                                             mode, STRATEGY)
            correct = res["letter"] == gt
            row = {
                "model": MODEL_LABEL, "strategy": STRATEGY,
                "question_id": record["question_id"],
                "sample_id": record.get("sample_id"),
                "source": source, "mode": mode,
                "topic": record["topic"], "city": record.get("city"),
                "image_mode": record.get("image_mode"),  # honest: benchmark recs carry none (null)
                "gt": gt, "letter": res["letter"], "correct": correct,
                "reasoning": res["reasoning"], "answer_text": res["answer_text"],
                "transcript": res["transcript"],
                "think_truncated": res["think_truncated"],
                "ctx_overflow": res["ctx_overflow"],
                "parse_attempts": res.get("parse_attempts", 1),
                "finish_reason": res["finish_reason"],
                "usage": res["usage"], "latency_s": res["latency_s"],
            }
            async with lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                all_rows.append(row)
                counters["done"] += 1
                counters[f"{mode}_correct"] += int(correct)
                counters[f"{mode}_n"] += 1
                if res["think_truncated"]:
                    counters["truncated"] += 1
                self_log(counters)
                if maybe_flush_summary_locked is not None:
                    maybe_flush_summary_locked()
        except Exception as e:  # one bad (record,mode) shouldn't kill the run
            async with lock:
                out_f.write(json.dumps({
                    "model": MODEL_LABEL, "strategy": STRATEGY,
                    "question_id": record.get("question_id"), "source": source,
                    "mode": mode, "error": repr(e),
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["errors"] += 1
        finally:
            queue.task_done()


def self_log(counters):
    if counters["done"] % 50 == 0 and counters["done"] > 0:
        def pct(m):
            n = counters[f"{m}_n"] or 1
            return 100 * counters[f"{m}_correct"] / n
        print(f"  [{STRATEGY}] {counters['done']} done | "
              f"full {pct('full'):.0f}% sat {pct('sat_only'):.0f}% "
              f"sv {pct('sv_only'):.0f}% blind {pct('blind'):.0f}% | "
              f"trunc {counters['truncated']} err {counters['errors']}", flush=True)


def write_summary(out_dir, rows, meta):
    s = summ.build_summary(rows, meta=meta)
    for name, payload in (("summary.json", json.dumps(s, indent=2, ensure_ascii=False)),
                          ("summary.md", summ.render_markdown(s))):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(payload)
        tmp.replace(out_dir / name)
    return s


async def summary_refresher(out_dir, all_rows, meta, counters, stop_evt, total,
                            t_start, done_at_start):
    while not stop_evt.is_set():
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=SUMMARY_EVERY_S)
        except asyncio.TimeoutError:
            pass
        done = counters["done"]
        elapsed = max(1e-6, time.monotonic() - t_start)
        gpm = round((done - done_at_start) / elapsed * 60, 1)
        remaining = max(0, total - done)
        eta_min = round(remaining / gpm, 1) if gpm > 0 else None
        meta.update({"updated": _now_iso(), "gens_per_min": gpm,
                     "eta": f"{eta_min} min" if eta_min is not None else None})
        write_summary(out_dir, list(all_rows), meta)


def _now_iso():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main_async(args):
    global MODEL_LABEL, SERVED_NAME, STRATEGY
    MODEL_LABEL = args.label
    SERVED_NAME = args.served_name
    STRATEGY = args.strategy

    data_root = Path(args.data_root)
    out_dir = RESULTS_DIR / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "ablation_log.jsonl")

    # Build worklist: one (source, record, mode) per applicable view, attribute
    # topics only, skipping records whose image files are missing for that view.
    recs = load_jsonl(args.data)
    if args.limit_per_topic:
        by_t = defaultdict(list)
        for r in recs:
            by_t[r.get("topic")].append(r)
        recs = [r for rs in by_t.values() for r in rs[: args.limit_per_topic]]

    source = "benchmark"
    work = []
    mode_counts = defaultdict(int)
    skipped_missing = 0
    for r in recs:
        topic = r.get("topic")
        if topic not in ATTR_TOPICS or topic in EXCLUDED_TOPICS:
            continue
        for mode in MODES:
            if args.modes and mode not in args.modes:
                continue
            if not has_required_images(r, data_root, mode):
                skipped_missing += 1
                continue
            work.append((source, r, mode))
            mode_counts[mode] += 1
    total_planned = len(work)

    # Resume — skip (source, qid, mode, strategy) already logged, preload prior
    # rows so the live summary reflects total progress.
    all_rows = summ.load_rows(log_path) if os.path.exists(log_path) else []
    all_rows = [r for r in all_rows if "error" not in r]
    done_keys = {(r.get("source"), r.get("question_id"), r.get("mode"), r.get("strategy"))
                 for r in all_rows if r.get("question_id") and r.get("mode")}
    if done_keys:
        before = len(work)
        work = [(s, r, m) for (s, r, m) in work
                if (s, r["question_id"], m, STRATEGY) not in done_keys]
        print(f"Resume: {before - len(work)} pairs already done, {len(work)} remain.",
              flush=True)

    print(f"[{args.strategy}] {len(work)} generations @ concurrency {args.concurrency}",
          flush=True)
    print(f"  mode breakdown: {dict(mode_counts)}  (skipped missing-image: {skipped_missing})",
          flush=True)
    print(f"  results dir: {out_dir}", flush=True)

    meta = {
        "model": args.label, "strategy": args.strategy,
        "model_name": args.model_name, "served_name": args.served_name,
        "base_url": args.base_url, "data": args.data, "sources": [source],
        "modes": args.modes or MODES,
        "think_budget": THINK_BUDGET, "answer_budget": ANSWER_BUDGET,
        "parse_retries": PARSE_RETRIES, "temperature": TEMPERATURE,
        "concurrency": args.concurrency, "total": total_planned,
        "already_done": total_planned - len(work), "started": _now_iso(),
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
    done_at_start = len(all_rows)

    def maybe_flush_summary_locked():
        if counters["done"] - counters["_summary_dirty_at"] >= SUMMARY_EVERY_N:
            counters["_summary_dirty_at"] = counters["done"]
            elapsed = max(1e-6, time.monotonic() - t_start)
            gpm = round(counters["done"] / elapsed * 60, 1)
            meta.update({"updated": _now_iso(), "gens_per_min": gpm,
                         "eta": f"{round(max(0,total_planned-counters['done'])/gpm,1)} min" if gpm else None})
            write_summary(out_dir, list(all_rows), meta)

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        refresher = asyncio.create_task(
            summary_refresher(out_dir, all_rows, meta, counters, stop_evt,
                              total_planned, t_start, done_at_start))
        workers = [
            asyncio.create_task(
                worker(f"w{i}", queue, session, args.base_url, data_root, out_f,
                       lock, counters, all_rows, maybe_flush_summary_locked))
            for i in range(args.concurrency)
        ]
        await queue.join()
        await asyncio.gather(*workers)
        stop_evt.set()
        await refresher
    out_f.close()

    meta["updated"] = meta["finished"] = _now_iso()
    final = write_summary(out_dir, summ.load_rows(log_path), meta)

    print(f"\n=== DONE [{args.strategy}] ===", flush=True)
    print(f"generations: {counters['done']} | truncated-think: {counters['truncated']} "
          f"| errors: {counters['errors']}")
    if final["progress"]["overall_acc"] is not None:
        print(f"overall acc: {final['progress']['overall_acc']}%")
    for m in MODES:
        if counters[f"{m}_n"]:
            print(f"  {m:9s}: {counters[f'{m}_correct']}/{counters[f'{m}_n']} correct")
    print(f"results: {out_dir}/  (ablation_log.jsonl, summary.json, summary.md)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_RUNNERS))
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--served-name", default="qwen9b")
    ap.add_argument("--model-name", default="qwen3.5-9b-awq")
    ap.add_argument("--label", default=None,
                    help="results/<label>/ ... defaults to the strategy name")
    ap.add_argument("--data", default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="dir image paths in the JSONL are relative to")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--modes", nargs="+", default=None, choices=MODES,
                    help="restrict to these views (default: all four)")
    ap.add_argument("--limit-per-topic", type=int, default=0,
                    help="cap records per topic (smoke testing)")
    args = ap.parse_args()
    if not args.label:
        args.label = args.strategy
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
