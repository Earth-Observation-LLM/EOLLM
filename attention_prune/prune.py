#!/usr/bin/env python3
"""prune.py — Phase 2: does the answer survive on the attended k-image subset?

Phase 1 (solve.py) solved each urban question on the full 4-angle street view and
recorded, on the CORRECT answer, how much the model attended to each SV image.
Phase 2 asks the redundancy question as a clean benchmark:

  Keep only the top-k attended images, re-ask in a SINGLE GREEDY PASS (no retries,
  no sampling, no steering), and measure what fraction of solved questions stay
  solved. That survival % is the headline — a per-question CEILING, reported as
  survival, never as a raw accuracy (the questions were selected for solvability).

Pool (locked with the user): solved_greedy ONLY. Every question entered Phase 2
having been solved deterministically on 4 images; survival is greedy-in/greedy-out,
so sampling is eliminated from BOTH ends.

Arms per k (the negative controls make the result falsifiable):
  top-k      keep the k MOST-attended images        (the claim)
  bottom-k   keep the k LEAST-attended images        (control: does ranking matter?)
  random-k   keep k images at a fixed seed           (control: does any k work?)
  fwd        k=1 only: always keep streetview_along_fwd, ignoring attention
             (tests "the forward image is THE important one" vs "attention adds
              something beyond a fixed position prior")
  zero       k=0: no images at all                    (leakage detector — a question
             that "survives" with zero images is text-solvable and is DROPPED from
             the headline)

The headline is the GAP (top-k − random-k / − bottom-k) with CIs, NOT top-k alone,
and it is computed on HEADLINE-ELIGIBLE questions only (neutral-prompt greedy wins;
already flagged per record by solve.py). Leaky topics (blind_baseline.py) are
reported with their blind baseline beside the survival, never pooled into the
headline. analyze() prints all of this; this script's job is to produce the
per-(question, arm) greedy verdicts efficiently and resumably.

Single greedy pass = pure benchmark. We REBUILD the prompt for the kept images via
suite_prompt.build_user_text(rec, kept_roles) so the image numbering the model sees
matches exactly the images present — no dangling "Image 3:" with no image 3.

Outputs (under output.dir/<model.key>/prune/):
  prune_results.jsonl   one line per (question_id, arm, k): kept roles, prediction,
                        correct(bool), gold
  prune_meta.json       provenance (model, signal, k list, pool, git sha, counts)

Run analysis after:  python prune.py --analyze        (no model load)

Usage (lab-ws SLURM, from attention_prune/):
  sbatch run_prune.slurm                      # full: k from config, all arms
  K=1 sbatch run_prune.slurm                  # k=1 only
  LIMIT=3 sbatch run_prune.slurm              # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUITE = REPO / "benchmark_suite"
for p in (SUITE, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import images as suite_images          # noqa: E402
import modes as suite_modes            # noqa: E402
import prompt as suite_prompt          # noqa: E402
import parsing as suite_parsing        # noqa: E402

FWD_ROLE = "streetview_along_fwd"

# Topics whose blind (zero-pixel) baseline is high enough that attention over
# images is meaningless for them — they cannot carry the headline. Kept in sync
# with blind_baseline.py / README (regenerate with `python blind_baseline.py`).
LEAKY_TOPICS = {"road_surface", "junction_type", "road_type",
                "urban_density", "land_use", "building_height"}
SAFE_TOPICS = {"amenity_richness", "transit_density"}


# ---------------------------------------------------------------------------
# Ranking: turn a solved record's stored attention into an ordering of SV roles
# ---------------------------------------------------------------------------

def rank_roles(attn: dict, signal: str, layer: int | None) -> list[str] | None:
    """Return the SV roles ordered MOST→LEAST attended.

    signal == "mean"   -> across-layer mean stored on attn["images"][i]["avg_norm_pct"]
    signal == "layer"  -> a single layer's per-image raw shares (normalized here)
    signal == "band"   -> mean of a contiguous layer band [layer, end] (late band)

    Answers are single-token in this data, so choice==avg; we use the avg side
    (identical numbers, honest name). Returns None if attention is unusable.
    """
    if not attn:
        return None
    imgs = attn.get("images") or []
    roles = [im.get("role") for im in imgs]
    if not roles or any(r is None for r in roles):
        return None

    if signal == "mean":
        shares = [float(im.get("avg_norm_pct", 0.0)) for im in imgs]
    else:
        per = attn.get("per_layer") or []
        if not per:
            return None
        if signal == "layer":
            if layer is None or layer < 0 or layer >= len(per):
                return None
            raw = [float(x) for x in per[layer].get("avg_raw_pct", [])]
        elif signal == "band":
            lo = layer if (layer is not None and layer >= 0) else int(0.75 * len(per))
            sel = per[lo:]
            if not sel:
                return None
            n = len(roles)
            raw = [0.0] * n
            for pl in sel:
                a = pl.get("avg_raw_pct", [])
                for i in range(min(n, len(a))):
                    raw[i] += float(a[i])
        else:
            raise ValueError(f"unknown signal {signal!r}")
        if len(raw) != len(roles):
            return None
        shares = raw

    # stable order: by share desc, ties broken by original index (deterministic)
    order = sorted(range(len(roles)), key=lambda i: (-shares[i], i))
    return [roles[i] for i in order]


def _stable_seed(rng_seed: int, qid: str, all_roles: list[str]) -> int:
    """A process-stable per-question seed (Python's str hash is salted, so we must
    NOT use hash()/__hash__ here or the 'deterministic' random subset changes run
    to run, breaking resume and reproducibility)."""
    import hashlib
    h = hashlib.sha256(f"{rng_seed}|{qid}|{','.join(all_roles)}".encode()).hexdigest()
    return int(h[:16], 16)


def parse_arm(arm: str):
    """Split an arm id into (base, seed). 'random:101' -> ('random', 101); a bare
    'random' -> ('random', None) meaning use the config default seed."""
    if ":" in arm:
        base, s = arm.split(":", 1)
        return base, int(s)
    return arm, None


def subset_for_arm(ranked: list[str], all_roles: list[str], arm: str, k: int,
                   rng_seed: int, qid: str = "") -> list[str] | None:
    """The roles to KEEP for a given arm at size k. ranked is most→least attended.

    Order of the returned roles follows all_roles (the canonical image order), so
    the rebuilt prompt numbers images consistently regardless of attention rank.
    `arm` may be 'random:SEED' to pick a specific random-control seed.
    Returns None if the arm is undefined for this k.
    """
    base, seed = parse_arm(arm)
    if base == "zero" or k == 0:
        return []
    if base == "fwd":
        if k != 1:
            return None
        return [FWD_ROLE] if FWD_ROLE in all_roles else None
    if k >= len(all_roles):
        keep = list(all_roles)  # nothing pruned
    elif base == "top":
        keep = ranked[:k]
    elif base == "bottom":
        keep = ranked[-k:]
    elif base == "random":
        import random
        r = random.Random(_stable_seed(seed if seed is not None else rng_seed,
                                       qid, all_roles))
        idx = list(range(len(all_roles)))
        r.shuffle(idx)
        keep = [all_roles[i] for i in idx[:k]]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if keep is None:
        return None
    keepset = set(keep)
    return [r for r in all_roles if r in keepset]   # canonical order


# ---------------------------------------------------------------------------
# Greedy single-pass runner (loads the model; lab-ws). Reuses solve.py's loader.
# ---------------------------------------------------------------------------

class GreedyRunner:
    """Minimal single-greedy-pass benchmark runner over a chosen image subset.

    Deliberately does NOT import the attention machinery — Phase 2 is a pure
    benchmark, no eager forward, no output_attentions. Reuses the same model load,
    chat-template (thinking off), max_pixels and parsing as Phase 1 so pixels and
    scoring are identical to solve.py and the main eval.
    """

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.g_cfg = cfg.get("generation", {})
        self.a_cfg = cfg.get("attention", {})
        self.max_new_tokens = int(self.g_cfg.get("max_new_tokens", 8))
        self.max_pixels = int(self.a_cfg.get("max_pixels", 1048576))
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        hf_id = self.model_cfg["hf_id"]
        local = REPO / hf_id
        if local.exists():
            hf_id = str(local)
        print(f"[load] {hf_id} (bf16, greedy benchmark, max_pixels={self.max_pixels})", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        # No eager attention needed for Phase 2 — use the default (faster) attn.
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        self.model.eval()
        tok = self._tok()
        if tok is not None:
            tok.padding_side = "left"
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
        self.pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        self._set_max_pixels()
        self._probe_thinking()

    def _tok(self):
        p = self.processor
        return p.tokenizer if hasattr(p, "tokenizer") else p

    def _set_max_pixels(self):
        ip = getattr(self.processor, "image_processor", None)
        if ip is None or not hasattr(ip, "size"):
            return
        size = ip.size
        try:
            size.longest_edge = self.max_pixels
            if getattr(size, "shortest_edge", None) and size.shortest_edge > self.max_pixels:
                size.shortest_edge = self.max_pixels
        except Exception as e:
            print(f"[warn] could not set max_pixels ({e})", flush=True)

    def _probe_thinking(self):
        """Match solve.py: thinking must be OFF for the benchmark too."""
        probe = [{"role": "user", "content": [{"type": "text", "text": "Reply A."}]}]
        try:
            self.processor.apply_chat_template(
                probe, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            self._takes_thinking = True
        except TypeError:
            self._takes_thinking = False
        print(f"[load] thinking OFF (template_takes_enable_thinking={self._takes_thinking})", flush=True)

    def _apply_template(self, msgs):
        if self._takes_thinking:
            return self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def predict(self, pil_images: list, user_text: str, options: dict) -> str:
        """One greedy pass; return the parsed letter (or '' if unparseable)."""
        import torch
        sys_prompt = suite_prompt.SYSTEM_PROMPT
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content":
                [{"type": "image"} for _ in pil_images]
                + [{"type": "text", "text": user_text}]},
        ]
        text = self._apply_template(msgs)
        if pil_images:
            inputs = self.processor([pil_images], [text], add_special_tokens=False,
                                    return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=[text], add_special_tokens=False,
                                    return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                use_cache=True, return_dict_in_generate=True, pad_token_id=self.pad_id)
        prompt_len = inputs["input_ids"].shape[1]
        ans_ids = gen.sequences[0, prompt_len:]
        raw = self._tok().decode(ans_ids, skip_special_tokens=True).strip()
        # parse-time <think> backstop (no-op when thinking is off)
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        parsed = suite_parsing.parse_answer(raw, options=options)
        return parsed.letter or ""

    def close(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Worklist: (solved_greedy record × arm × k) -> a greedy pass over kept images
# ---------------------------------------------------------------------------

def _arms_for(k: int, arms_cfg: list[str], random_seeds: list[int]) -> list[str]:
    """The concrete arm ids defined at this k.

    'fwd' only at k=1; 'zero' is added separately as its own k=0 pass. The
    'random' control is expanded into one arm per seed ('random:3407', ...) so
    each seed is an independent, separately-resumable draw the analysis can pool.
    A k that keeps all images makes the random control degenerate (it would equal
    top/bottom), so we collapse the random seeds to a single arm there.
    """
    out = []
    for a in arms_cfg:
        if a == "fwd" and k != 1:
            continue
        if a == "zero":
            continue
        if a == "random":
            seeds = random_seeds if k < 4 else random_seeds[:1]  # k>=4 = no pruning
            out.extend(f"random:{s}" for s in seeds)
        else:
            out.append(a)
    return out


def build_tasks(solved_records, image_root, max_edge, signal, layer,
                ks, arms_cfg, random_seeds, headline_only):
    """Materialize the (record, arm, k, kept_roles, pil_subset, user_text) tasks.

    Images are built ONCE per record (full 4 SV angles) and sliced per arm, so we
    pay the pixel work once. user_text is rebuilt per subset so image numbering
    matches the kept images.
    """
    tasks = []
    skipped = Counter()
    want_zero = "zero" in arms_cfg

    for rec_wrap in solved_records:
        rec = rec_wrap["record"]
        if headline_only and not rec_wrap.get("headline_eligible"):
            skipped["not_headline"] += 1
            continue
        attn = rec_wrap.get("attention")
        ranked = rank_roles(attn, signal, layer)
        if ranked is None:
            skipped["no_rank"] += 1
            continue
        try:
            full = suite_images.build_images(rec, image_root, max_edge)
        except Exception as e:
            skipped["build_fail"] += 1
            if skipped["build_fail"] <= 5:
                print(f"  [skip] {rec.get('question_id')}: {type(e).__name__}: {e}", flush=True)
            continue
        sv = suite_modes.filter_images_for_mode(full, "sv_only")
        if not sv:
            skipped["no_sv"] += 1
            continue
        all_roles = [s["role"] for s in sv]
        by_role = {s["role"]: s["image"] for s in sv}
        gold = rec_wrap.get("gold") or suite_parsing.parse_letter(rec.get("answer"))

        qid = rec.get("question_id")
        for k in ks:
            for arm in _arms_for(k, arms_cfg, random_seeds):
                keep = subset_for_arm(ranked, all_roles, arm, k,
                                      random_seeds[0], qid=qid)
                if keep is None:
                    continue
                pil = [by_role[r] for r in keep]
                text = suite_prompt.build_user_text(rec, keep)
                tasks.append({
                    "question_id": qid, "topic": rec_wrap["topic"],
                    "city": rec.get("city"), "gold": gold, "options": rec.get("options"),
                    "arm": arm, "k": k, "kept_roles": keep,
                    "pil": pil, "user_text": text,
                })
        if want_zero:
            text0 = suite_prompt.build_user_text(rec, [])
            tasks.append({
                "question_id": rec.get("question_id"), "topic": rec_wrap["topic"],
                "city": rec.get("city"), "gold": gold, "options": rec.get("options"),
                "arm": "zero", "k": 0, "kept_roles": [],
                "pil": [], "user_text": text0,
            })

    if skipped:
        print(f"  [worklist] skipped: {dict(skipped)}", flush=True)
    return tasks


# ---------------------------------------------------------------------------
# Analysis (no model) — survival %, Wilson CIs, gaps, blind baseline beside
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _blind_baselines(data_path: Path):
    """Per-topic zero-pixel modal-option baseline (same logic as blind_baseline.py)."""
    recs = [json.loads(l) for l in open(data_path) if l.strip()]
    by_topic = defaultdict(list)
    for r in recs:
        if r.get("topic") in (SAFE_TOPICS | LEAKY_TOPICS):
            by_topic[r["topic"]].append(r)
    out = {}
    for topic, rs in by_topic.items():
        counts = Counter()
        for r in rs:
            ans = (r.get("answer") or "").strip().upper()
            letter = ans[0] if ans and ans[0] in "ABCD" else None
            opts = r.get("options") or {}
            gt = opts.get(letter) if letter else None
            if gt is not None:
                counts[gt] += 1
        if rs and counts:
            out[topic] = counts.most_common(1)[0][1] / len(rs)
    return out


def analyze(out_dir: Path, data_path: Path):
    rp = out_dir / "prune_results.jsonl"
    if not rp.exists():
        raise SystemExit(f"no prune_results.jsonl in {out_dir} — run the benchmark first")
    rows = [json.loads(l) for l in open(rp) if l.strip()]
    blind = _blind_baselines(data_path)

    # survival[(topic, base_arm, k)] = (n_correct, n_total); the per-seed random
    # arms ('random:3407', ...) are POOLED into one 'random' estimate. We also keep
    # per-seed survival to report the spread across seeds (robustness of the control).
    surv = defaultdict(lambda: [0, 0])
    seed_surv = defaultdict(lambda: [0, 0])   # (topic, seed, k) -> for random arms
    ks, bases, rand_seeds = set(), set(), set()
    for r in rows:
        base, seed = parse_arm(r["arm"])
        key = (r["topic"], base, r["k"])
        surv[key][1] += 1
        surv[key][0] += int(r["correct"])
        if base == "random" and seed is not None:
            sk = (r["topic"], seed, r["k"])
            seed_surv[sk][1] += 1
            seed_surv[sk][0] += int(r["correct"])
            rand_seeds.add(seed)
        ks.add(r["k"])
        bases.add(base)
    ks = sorted(ks)
    rand_seeds = sorted(rand_seeds)
    arm_order = [a for a in ["top", "fwd", "random", "bottom", "zero"] if a in bases]

    def cell(topic, arm, k):
        c, n = surv.get((topic, arm, k), [0, 0])
        return c, n

    print("\n" + "=" * 100)
    print("PHASE 2 — SURVIVAL of solved_greedy questions on pruned SV subsets")
    print("survival = of questions solved on 4 images, %% still solved on the kept subset")
    print("=" * 100)

    all_topics = sorted({r["topic"] for r in rows})
    for k in ks:
        if k == 0:
            continue
        print(f"\n### k = {k} images kept")
        hdr = f"{'topic':18s} {'n':>5s} {'blind':>6s} " + " ".join(f"{a:>16s}" for a in arm_order if a != 'zero')
        print(hdr)
        print("-" * len(hdr))
        for topic in all_topics:
            # n is the same across arms at this k (same question pool); take 'top'
            _, n = cell(topic, "top", k)
            if n == 0:
                continue
            bl = blind.get(topic)
            bls = f"{bl:.2f}" if bl is not None else "  -"
            parts = []
            for a in arm_order:
                if a == "zero":
                    continue
                c, na = cell(topic, a, k)
                if na == 0:
                    parts.append(f"{'-':>16s}")
                    continue
                p, lo, hi = wilson(c, na)
                parts.append(f"{100*p:5.1f}% [{100*lo:4.0f},{100*hi:4.0f}]")
            tag = " (LEAKY)" if topic in LEAKY_TOPICS else (" (safe)" if topic in SAFE_TOPICS else "")
            print(f"{topic:18s} {n:5d} {bls:>6s} " + " ".join(parts) + tag)

        # headline pooled over SAFE topics only, with the gap
        for label, tset in [("SAFE-pool (headline)", SAFE_TOPICS)]:
            tc = ta = bc = ba = rc = ra = 0
            for topic in tset:
                c, n = cell(topic, "top", k); tc += c; ta += n
                c, n = cell(topic, "bottom", k); bc += c; ba += n
                c, n = cell(topic, "random", k); rc += c; ra += n
            if ta:
                pt, lt, ht = wilson(tc, ta)
                line = f"\n  {label}: top-{k} survival {100*pt:.1f}% [{100*lt:.0f},{100*ht:.0f}] (n={ta})"
                if ra:
                    pr, lr, hr = wilson(rc, ra)
                    line += f"  | random-{k} {100*pr:.1f}% (pooled over {len(rand_seeds) or 1} seeds)"
                    line += f"  | gap {100*(pt-pr):+.1f}pp"
                if ba:
                    pb, _, _ = wilson(bc, ba)
                    line += f"  | bottom-{k} {100*pb:.1f}%"
                print(line)
                # per-seed spread of the random control over the SAFE pool — shows
                # the gap isn't an artifact of one lucky/unlucky random draw.
                if len(rand_seeds) > 1:
                    seed_pcts = []
                    for s in rand_seeds:
                        sc = sn = 0
                        for topic in tset:
                            c, n = seed_surv.get((topic, s, k), [0, 0])
                            sc += c; sn += n
                        if sn:
                            seed_pcts.append(100 * sc / sn)
                    if seed_pcts:
                        lo_s, hi_s = min(seed_pcts), max(seed_pcts)
                        print(f"    random-{k} per-seed: "
                              + " ".join(f"{p:.1f}%" for p in seed_pcts)
                              + f"  (spread {hi_s-lo_s:.1f}pp)")

    # k=0 leakage detector
    if 0 in ks:
        print("\n### k = 0 (no images) — LEAKAGE DETECTOR")
        print("a question that survives here is text-solvable; it inflates every arm")
        for topic in all_topics:
            c, n = cell(topic, "zero", 0)
            if n:
                p, lo, hi = wilson(c, n)
                print(f"  {topic:18s} n={n:4d}  zero-image survival {100*p:5.1f}% [{100*lo:.0f},{100*hi:.0f}]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _resolve_data(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    cand = REPO / rel
    if cand.exists():
        return cand
    dd = os.environ.get("DATASET_DIR")
    if dd:
        marker = "EODATA_compressed_final/"
        tail = rel.split(marker, 1)[1] if marker in rel else Path(rel).name
        remapped = Path(dd) / tail
        if remapped.exists():
            return remapped
    return cand


def _load_solved(solved_path: Path, data_path: Path, pool: set[str]):
    """Phase-1 solved records in the chosen pool (default solved_greedy), joined to
    the ORIGINAL benchmark records by question_id.

    build_images needs the record's real `images` dict (relative paths) and
    `image_mode` — exactly the fields solve.py used in Phase 1. Rather than
    reconstruct those from the stored attention output (lossy: no satellite path,
    absolute lab-ws paths), we re-read the source benchmark and look each solved
    question up by id. This guarantees Phase-2 pixels == Phase-1 pixels.
    """
    src = {}
    for line in open(data_path):
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = rec.get("question_id")
        if qid is not None:
            src[qid] = rec

    out, missing = [], 0
    for line in open(solved_path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") not in pool or not r.get("attention"):
            continue
        qid = r.get("question_id")
        rec = src.get(qid)
        if rec is None:
            missing += 1
            continue
        out.append({"record": rec, "topic": r.get("topic"), "gold": r.get("gold"),
                    "headline_eligible": r.get("headline_eligible", False),
                    "attention": r.get("attention")})
    if missing:
        print(f"  [warn] {missing} solved questions not found in source benchmark", flush=True)
    return out


def _load_done(out_dir: Path):
    done = set()
    p = out_dir / "prune_results.jsonl"
    if p.exists():
        for line in open(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get("question_id"), r.get("arm"), r.get("k")))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--analyze", action="store_true", help="analysis only (no model)")
    ap.add_argument("--limit", type=int, default=None, help="cap solved records (smoke)")
    ap.add_argument("--ks", type=int, nargs="+", default=None, help="override k list")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg.get("prune", {}) or {}
    signal = pcfg.get("signal", "mean")
    layer = pcfg.get("layer")
    arms_cfg = pcfg.get("arms", ["top", "fwd", "random", "bottom", "zero"])
    # Multiple seeds for the random control → the random arm is one independent draw
    # per seed; the analysis pools them and reports the per-seed spread. Back-compat:
    # a single 'random_seed' is honored if 'random_seeds' is absent.
    random_seeds = pcfg.get("random_seeds")
    if not random_seeds:
        random_seeds = [int(pcfg.get("random_seed", 3407))]
    random_seeds = [int(s) for s in random_seeds]
    headline_only = bool(pcfg.get("headline_only", False))
    pool = set(pcfg.get("pool", ["solved_greedy"]))
    ks = args.ks or pcfg.get("ks", [1])
    # env overrides (slurm convenience)
    if os.environ.get("K"):
        ks = [int(x) for x in os.environ["K"].split()]

    ds = cfg["dataset"]
    data_path = _resolve_data(ds["path"])
    image_root = _resolve_data(ds["image_root"])
    out_dir = (REPO / cfg["output"]["dir"] / cfg["model"]["key"] / "prune")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        analyze(out_dir, data_path)
        return

    solved_path = (REPO / cfg["output"]["dir"] / cfg["model"]["key"] / "solved.jsonl")
    if not solved_path.exists():
        raise SystemExit(f"no Phase-1 solved.jsonl at {solved_path}; run solve.py first")
    solved = _load_solved(solved_path, data_path, pool)
    if args.limit:
        solved = solved[:args.limit]
    print(f"pool={sorted(pool)} headline_only={headline_only} -> {len(solved)} Phase-1 records",
          flush=True)
    print(f"signal={signal} layer={layer} ks={ks} arms={arms_cfg} "
          f"random_seeds={random_seeds}", flush=True)

    max_edge = int(cfg.get("attention", {}).get("image_max_edge", 768))
    tasks = build_tasks(solved, image_root, max_edge, signal, layer, ks,
                        arms_cfg, random_seeds, headline_only)
    print(f"tasks: {len(tasks)}  per-arm {dict(Counter(t['arm'] for t in tasks))}", flush=True)

    done = _load_done(out_dir)
    if done:
        before = len(tasks)
        tasks = [t for t in tasks if (t["question_id"], t["arm"], t["k"]) not in done]
        print(f"resume: {len(done)} done, {len(tasks)}/{before} remaining", flush=True)

    runner = GreedyRunner(cfg)
    res_f = open(out_dir / "prune_results.jsonl", "a")
    prog = out_dir / "prune_progress.json"
    t0 = time.time()
    counts = Counter()
    for n, t in enumerate(tasks, 1):
        pred = runner.predict(t["pil"], t["user_text"], t["options"])
        correct = bool(pred) and pred == t["gold"]
        counts[f"{t['arm']}_k{t['k']}"] += int(correct)
        rec = {"question_id": t["question_id"], "topic": t["topic"], "city": t["city"],
               "arm": t["arm"], "k": t["k"], "kept_roles": t["kept_roles"],
               "gold": t["gold"], "prediction": pred, "correct": correct}
        res_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res_f.flush()
        if n % 10 == 0 or n == len(tasks):
            el = time.time() - t0
            rate = el / n
            eta = rate * (len(tasks) - n) / 60
            prog.write_text(json.dumps({
                "done": n, "total": len(tasks),
                "pct": round(100 * n / len(tasks), 1) if tasks else 100.0,
                "rate_s_per_pass": round(rate, 2), "eta_min": round(eta, 1),
                "survivors_by_arm_k": dict(counts),
            }, indent=2))
            print(f"  [{n}/{len(tasks)}] {el/60:.1f} min, {rate:.1f}s/pass, ETA {eta:.0f} min",
                  flush=True)
    runner.close()
    res_f.close()

    meta = {"model": cfg["model"], "signal": signal, "layer": layer, "ks": ks,
            "arms": arms_cfg, "pool": sorted(pool), "headline_only": headline_only,
            "random_seeds": random_seeds, "n_tasks": len(tasks),
            "dataset": {"path": str(data_path), "image_root": str(image_root)},
            "git_sha": _git_sha(), "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "prune_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\ndone -> {out_dir}\nrun: python prune.py --analyze", flush=True)


if __name__ == "__main__":
    main()
