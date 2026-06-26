#!/usr/bin/env python3
"""solve.py — Phase 1 collector: solve each urban question, capture per-image
attention on the CORRECT answer.

Two phases per question (street-view-only, 4 angle images):

  PHASE A — solve (fast generation, no attention)
    Walk the ladder (config.ladder): greedy first, then steered, then rising
    temperature with the per-task prompt variants. Freeze on the FIRST correct
    letter and record the exact winning settings (temp, seed, variant).

  PHASE B — attend (one eager forward on the winning attempt)
    Re-run the model in attn_implementation="eager" reproducing the winning
    generation, and read off how much the answer attends to each of the 4 SV
    image spans (both the choice-token and answer-averaged shares, per layer).
      - greedy win  -> deterministic; the forward reproduces gold exactly.
      - sampled win -> re-sample with the saved seed up to N times until the
        correct letter reappears, THEN read attention. If it never reappears the
        question is `solved_unstable` (kept, but excluded from pruning).

Outputs (under output.dir/<model.key>/):
  solved.jsonl    one line per SOLVED question, with the attention block
  unsolved.jsonl  questions never answered correctly across the ladder
  meta.json       provenance (model, ladder, dataset, git sha, counts)

Reuses benchmark_suite for pixel-identical images / sv_only ablation / parsing /
prompt scaffolding, so this never drifts from the main eval.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
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
import prompts as task_prompts         # noqa: E402

import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove a <think>...</think> block (and any dangling open tag) before the
    answer is parsed. Backstop only — thinking is verified off at load."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text.lower():            # unterminated (truncated by max_new_tokens)
        text = _re.split(r"(?i)<think>", text)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Vision-span token-id auto-detection (do NOT hardcode Qwen3.5's ids)
# ---------------------------------------------------------------------------

def detect_vision_ids(processor, model_cfg) -> tuple[int, int]:
    """Find the <|vision_start|> / <|vision_end|> token ids for THIS model.

    Qwen3.6 may differ from 3.5. We resolve via the tokenizer's special-token
    vocabulary; config overrides win if present.
    """
    if model_cfg.get("vision_start_id") and model_cfg.get("vision_end_id"):
        return int(model_cfg["vision_start_id"]), int(model_cfg["vision_end_id"])
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    start = tok.convert_tokens_to_ids("<|vision_start|>")
    end = tok.convert_tokens_to_ids("<|vision_end|>")
    unk = getattr(tok, "unk_token_id", None)
    if start is None or end is None or start == unk or end == unk or start < 0 or end < 0:
        raise RuntimeError(
            "could not auto-detect <|vision_start|>/<|vision_end|> ids; set "
            "model.vision_start_id / model.vision_end_id in config.yaml")
    return int(start), int(end)


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------

class Collector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.ladder = cfg["ladder"]
        self.a_cfg = cfg.get("attention", {})
        self.g_cfg = cfg.get("generation", {})
        self.max_new_tokens = int(self.g_cfg.get("max_new_tokens", 8))
        self.base_seed = int(self.g_cfg.get("base_seed", 3407))
        self.max_pixels = int(self.a_cfg.get("max_pixels", 1048576))
        self.layers_spec = self.a_cfg.get("layers", "all")
        self.store_per_layer = bool(self.a_cfg.get("store_per_layer", True))
        self._load()

    def _load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        hf_id = self.model_cfg["hf_id"]
        # repo-relative local path?
        local = REPO / hf_id
        if local.exists():
            hf_id = str(local)
        print(f"[load] {hf_id} (bf16, eager, max_pixels={self.max_pixels})", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", trust_remote_code=True)
        self.model.eval()
        tok = self._tok()
        if tok is not None:
            tok.padding_side = "left"
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
        self.pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        self.eos_id = getattr(tok, "eos_token_id", None)
        self._set_max_pixels()
        self.vstart, self.vend = detect_vision_ids(self.processor, self.model_cfg)
        print(f"[load] vision ids: start={self.vstart} end={self.vend}", flush=True)
        cfg = self.model.config
        tc = getattr(cfg, "text_config", None)
        self.n_layers = (getattr(cfg, "num_hidden_layers", None)
                         or (getattr(tc, "num_hidden_layers", None) if tc else None)
                         or 48)
        self._verify_thinking_off()

    def _verify_thinking_off(self):
        """Make 'thinking off' real, not just requested. (1) Does the chat
        template accept enable_thinking? (2) Render a probe turn and assert the
        generation prompt does NOT invite open-ended reasoning. Fails loudly.

        NB: Qwen renders a CLOSED, EMPTY `<think>\\n\\n</think>` block when thinking
        is off — that is the correct thinking-off priming (the model is told the
        reasoning slot is already finished), NOT thinking being on. We only reject
        a think block that is left OPEN (no closing tag) or non-empty, which is what
        an actually-thinking template emits."""
        probe = [{"role": "user", "content": [{"type": "text", "text": "Reply A."}]}]
        # Empirically test the kwarg (some processors accept **kwargs silently).
        try:
            self.processor.apply_chat_template(
                probe, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            takes = True
        except TypeError:
            takes = False
        self._template_takes_thinking = takes
        rendered = self.processor.apply_chat_template(
            probe, add_generation_prompt=True, tokenize=False,
            **({"enable_thinking": False} if takes else {}))
        tail = rendered[rendered.rfind("<|im_start|>assistant"):] if "<|im_start|>assistant" in rendered else rendered
        opens = tail.count("<think>")
        closes = tail.count("</think>")
        # extract any think-block contents in the assistant turn
        import re
        nonempty = any(m.strip() for m in re.findall(r"<think>(.*?)</think>", tail, re.DOTALL))
        if opens > closes or nonempty:
            raise RuntimeError(
                "thinking appears ENABLED: the generation prompt has an open or "
                "non-empty <think> block even with enable_thinking=False. Refusing "
                "to run — set the model's chat template to non-thinking before "
                f"collecting attention. (rendered tail: {tail[-120:]!r})")
        print(f"[load] thinking OFF (template_takes_enable_thinking={takes})", flush=True)

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

    # -- prompt / inputs ---------------------------------------------------

    def _messages(self, item, steer: str):
        sys_prompt = suite_prompt.SYSTEM_PROMPT + (("\n\n" + steer) if steer else "")
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content":
                [{"type": "image"} for _ in item["images"]]
                + [{"type": "text", "text": item["user_text"]}]},
        ]

    def _apply_template(self, item, steer: str) -> str:
        msgs = self._messages(item, steer)
        if self._template_takes_thinking:
            return self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        # The signature does not accept enable_thinking. That is only safe if the
        # template emits NO thinking block by default (verified once at load); the
        # parse-time <think> strip is a further backstop.
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

    def _make_inputs(self, item, steer: str):
        text = self._apply_template(item, steer)
        if item["images"]:
            return self.processor([item["images"]], [text], add_special_tokens=False,
                                  return_tensors="pt").to(self.model.device)
        return self.processor(text=[text], add_special_tokens=False,
                              return_tensors="pt").to(self.model.device)

    # -- phase A: fast solve ----------------------------------------------

    @torch.no_grad()
    def _generate(self, inputs, temp: float, seed: int | None):
        """Generate the answer; return BOTH the decoded text and the exact
        generated ids. The ids (not a re-tokenization of the text) are what the
        attention forward must append — decode->re-encode is not identity and
        would shift the choice token the attention is read at."""
        do_sample = temp > 0.0
        if do_sample and seed is not None:
            torch.manual_seed(seed)
        kwargs = dict(max_new_tokens=self.max_new_tokens, use_cache=True,
                      return_dict_in_generate=True, pad_token_id=self.pad_id)
        if do_sample:
            kwargs.update(do_sample=True, temperature=temp)
        else:
            kwargs.update(do_sample=False)
        gen = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        ans_ids = gen.sequences[0, prompt_len:]          # exact generated ids
        # Backstop: if the model ever emits a <think>...</think> block despite the
        # template, drop it BEFORE the answer is parsed. NOTE: ans_ids (used for the
        # attention forward) are left intact — the attention is read over the actual
        # generated tokens, and with thinking verified off at load this should be a
        # no-op in practice; the strip only affects letter PARSING below.
        tok = self._tok()
        raw = tok.decode(ans_ids, skip_special_tokens=True).strip()
        raw = _strip_think(raw)
        return raw, ans_ids

    def solve(self, item) -> dict | None:
        """Walk the ladder; return the winning attempt dict or None if unsolved."""
        gold = suite_parsing.parse_letter(item["record"].get("answer"))
        variants = {v.name: v for v in task_prompts.variants_for(item["topic"])}
        attempt_idx = 0
        for rung in self.ladder:
            temp = float(rung["temp"])
            samples = int(rung.get("samples", 1))
            for vname in rung["variants"]:
                v = variants.get(vname)
                if v is None:
                    continue
                inputs = self._make_inputs(item, v.steer)
                for s in range(samples):
                    seed = self.base_seed + attempt_idx if temp > 0 else None
                    raw, ans_ids = self._generate(inputs, temp, seed)
                    parsed = suite_parsing.parse_answer(raw, options=item["record"]["options"])
                    attempt_idx += 1
                    if parsed.letter is not None and gold is not None and parsed.letter == gold:
                        # Keep the winning inputs + exact generated ids so the
                        # attention forward reuses THEM (no regeneration, no
                        # re-tokenization) — the attention is then read over the
                        # very tokens that were generated.
                        return {"temp": temp, "variant": vname, "steer": v.steer,
                                "seed": seed, "raw": raw, "letter": parsed.letter,
                                "n_attempts": attempt_idx, "greedy": temp == 0.0,
                                "inputs": inputs, "ans_ids": ans_ids}
        return None

    # -- phase B: attention on the winning attempt -------------------------

    @staticmethod
    def _detect_spans(ids_row, vstart, vend):
        ids = ids_row.tolist()
        spans, i = [], 0
        while i < len(ids):
            if ids[i] == vstart:
                j = i + 1
                while j < len(ids) and ids[j] != vend:
                    j += 1
                spans.append((i + 1, j))
                i = j + 1
            else:
                i += 1
        return spans

    def _layers_index(self, n_returned: int):
        n = min(self.n_layers, n_returned)
        spec = self.layers_spec
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            lo, hi = spec
            idx = list(range(max(0, lo), min(n, hi)))
        elif spec == "last_25pct":
            idx = list(range(int(self.n_layers * 0.75), self.n_layers))
        else:
            idx = list(range(self.n_layers))
        idx = [li for li in idx if li < n_returned]
        return idx or list(range(n_returned))

    @torch.no_grad()
    def _attention_forward(self, item, win):
        """One eager forward over [winning prompt + the EXACT generated answer ids]
        -> per-image attention. No regeneration and no re-tokenization: it reuses
        `win["inputs"]` and `win["ans_ids"]` captured at solve time, so the
        attention is read over the very tokens that produced the gold letter.
        Deterministic regardless of the winning temperature (the sampling already
        happened; this is a forced teacher-forcing forward over the kept ids)."""
        tok = self._tok()
        inputs = win["inputs"]
        target_letter = win["letter"]
        prompt_len = inputs["input_ids"].shape[1]
        keep = [t for t in win["ans_ids"].tolist() if t not in (self.pad_id, self.eos_id)]
        if not keep:
            return None

        ans_t = torch.tensor(keep, device=self.model.device).unsqueeze(0)
        full_ids = torch.cat([inputs["input_ids"], ans_t], dim=1)
        attn_mask = torch.ones_like(full_ids)
        fwd = {"input_ids": full_ids, "attention_mask": attn_mask,
               "output_attentions": True, "use_cache": False}
        model_dtype = next(self.model.parameters()).dtype
        if "mm_token_type_ids" in inputs:
            mm = inputs["mm_token_type_ids"]
            pad = torch.zeros((mm.shape[0], ans_t.shape[1]), dtype=mm.dtype, device=mm.device)
            fwd["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs:
                v = inputs[k]
                if k == "pixel_values" and v.is_floating_point():
                    v = v.to(model_dtype)
                fwd[k] = v
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            out = self.model(**fwd)
        attentions = out.attentions

        L = full_ids.shape[1]
        spans = self._detect_spans(full_ids[0], self.vstart, self.vend)
        spans_ok = (len(spans) == len(item["images"]))
        q_start = prompt_len - 1
        # answer token i is produced at query position prompt_len-1+i; the last
        # answer token has no "next" to predict, so rows prompt_len-1 .. L-2.
        q_rows = list(range(q_start, L - 1))
        choice_local = self._find_choice_index(keep, tok, target_letter)
        choice_row = q_start + choice_local if choice_local is not None else q_rows[0]

        layer_idx = self._layers_index(len(attentions))
        per_layer = self._compute_per_layer(attentions, layer_idx, q_rows, choice_row, spans)
        block = self._assemble(per_layer, item["roles"], len(item["images"]),
                               target_letter, choice_local, spans_ok, layer_idx)
        return block

    @staticmethod
    def _find_choice_index(answer_ids, tok, letter):
        for i, tid in enumerate(answer_ids):
            if tok.decode([tid]).strip().upper() == letter:
                return i
        return None

    def _compute_per_layer(self, attentions, layer_idx, q_rows, choice_row, spans):
        results = []
        for li in layer_idx:
            A = attentions[li][0].float().mean(dim=0)   # head-avg -> [L, L]
            rows_avg = A[q_rows, :].mean(dim=0)
            rows_choice = A[choice_row, :]
            per_avg = [float(rows_avg[s:e].sum()) for (s, e) in spans]
            per_choice = [float(rows_choice[s:e].sum()) for (s, e) in spans]
            results.append({"layer": li, "avg_raw": per_avg, "choice_raw": per_choice,
                            "row_sum_check": float(rows_avg.sum())})
        return results

    def _assemble(self, per_layer, roles, n_images, letter, choice_local, spans_ok, layer_idx):
        import statistics as st

        def mean_layers(key, k):
            return st.fmean(pl[key][k] for pl in per_layer) if per_layer else 0.0

        avg_raw = [mean_layers("avg_raw", k) for k in range(n_images)]
        choice_raw = [mean_layers("choice_raw", k) for k in range(n_images)]
        tot_avg = sum(avg_raw) or 1e-9
        tot_choice = sum(choice_raw) or 1e-9
        images = []
        for k in range(n_images):
            images.append({
                "image": k + 1,
                "role": roles[k] if k < len(roles) else f"image_{k+1}",
                "avg_raw_pct": round(100 * avg_raw[k], 4),
                "avg_norm_pct": round(100 * avg_raw[k] / tot_avg, 4),
                "choice_raw_pct": round(100 * choice_raw[k], 4),
                "choice_norm_pct": round(100 * choice_raw[k] / tot_choice, 4),
            })
        block = {
            "n_images": n_images, "choice_letter": letter,
            "choice_token_index": choice_local,
            "layers_used": ("all" if self.layers_spec == "all" else self.layers_spec),
            "n_layers_used": len(layer_idx),
            "image_attention_total_avg_pct": round(100 * sum(avg_raw), 4),
            "spans_match_images": spans_ok,
            "row_softmax_check": round(st.fmean(pl["row_sum_check"] for pl in per_layer), 4) if per_layer else None,
            "images": images,
        }
        if self.store_per_layer:
            block["per_layer"] = [
                {"layer": pl["layer"],
                 "avg_raw_pct": [round(100 * x, 4) for x in pl["avg_raw"]],
                 "choice_raw_pct": [round(100 * x, 4) for x in pl["choice_raw"]]}
                for pl in per_layer]
        return block

    # -- per-question driver ----------------------------------------------

    def process(self, item) -> dict:
        rec = item["record"]
        gold = suite_parsing.parse_letter(rec.get("answer"))
        base = {"question_id": rec.get("question_id"), "sample_id": rec.get("sample_id"),
                "topic": item["topic"], "city": rec.get("city"),
                "question": rec.get("question"), "options": rec.get("options"),
                "gold": gold, "image_roles": item["roles"],
                "image_paths": item["paths"]}

        win = self.solve(item)
        if win is None:
            return {**base, "status": "unsolved", "attention": None,
                    "headline_eligible": False}

        # The win carries the EXACT inputs + generated ids that produced gold, so
        # the attention forward is over those very tokens — deterministic whether
        # the win was greedy or sampled (the sampling already happened). The old
        # "reproduce the seed" dance is gone: there is nothing to reproduce.
        block = self._attention_forward(item, win)
        if block is None:
            status = "solved_no_attention"   # empty answer / no usable tokens
            attn = None
        elif win["greedy"]:
            status = "solved_greedy"
            attn = block
        else:
            status = "solved_sampled"
            attn = block

        # Headline-eligibility (per the validity review): the headline survival
        # curve uses only NEUTRAL-prompt, GREEDY wins with a usable attention map.
        # Steered wins and sampled wins are recorded and shown, but as clearly
        # labeled secondary curves — a steered attention map is partly an artifact
        # of the instruction, and a sampled win is off-distribution from greedy.
        headline_eligible = (status == "solved_greedy"
                             and win["variant"] == "neutral"
                             and block.get("spans_match_images", False))

        return {**base, "prediction": win["letter"], "status": status,
                "headline_eligible": headline_eligible,
                "winning": {k: win[k] for k in ("temp", "variant", "seed", "n_attempts", "greedy")},
                "raw_response": win["raw"], "attention": attn}

    def close(self):
        import gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Worklist (sv_only, 8 urban topics) — reuses suite image construction
# ---------------------------------------------------------------------------

def build_worklist(records, image_root, image_max_edge, topics, limit_per_topic):
    by_topic = defaultdict(list)
    for r in records:
        if r.get("topic") in topics:
            by_topic[r["topic"]].append(r)
    work, skipped = [], 0
    for topic, recs in by_topic.items():
        if limit_per_topic:
            recs = recs[:limit_per_topic]
        for rec in recs:
            try:
                full = suite_images.build_images(rec, image_root, image_max_edge)
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  [skip] {rec.get('question_id')}: {type(e).__name__}: {e}", flush=True)
                continue
            sel = suite_modes.filter_images_for_mode(full, "sv_only")
            if not sel:
                skipped += 1
                continue
            roles = [s["role"] for s in sel]
            pil = [s["image"] for s in sel]
            text = suite_prompt.build_user_text(rec, roles)
            # record image file paths (for the viewer) parallel to roles
            paths = _sv_paths_for(rec, roles, image_root)
            work.append({"record": rec, "topic": topic, "images": pil,
                         "roles": roles, "user_text": text, "paths": paths})
    if skipped:
        print(f"  [worklist] skipped {skipped} records", flush=True)
    return work


def _sv_paths_for(rec, roles, image_root):
    """Best-effort source file path per SV role, for the viewer thumbnails."""
    imgs = rec.get("images", {}) or {}
    out = []
    for role in roles:
        # roles look like streetview_<angle>
        angle = role.replace("streetview_", "")
        rel = imgs.get(f"streetview_{angle}")
        out.append(str(Path(image_root) / rel) if rel else None)
    return out


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _load_done(out_dir: Path):
    """Question ids already recorded + their status counts, for resume."""
    done, counts = set(), Counter()
    for name in ("solved.jsonl", "unsolved.jsonl"):
        p = out_dir / name
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written trailing line from a kill
            qid = r.get("question_id")
            if qid is not None:
                done.add(qid)
                counts[r.get("status", "?")] += 1
    return done, counts


def _write_progress(path: Path, done, total, counts, rate_s, eta_min, model_key):
    """A small heartbeat file you can poll (status.py / scp) without tailing logs."""
    headline = counts.get("solved_greedy", 0)  # neutral-greedy subset is computed in analysis;
    solved = sum(v for k, v in counts.items() if k.startswith("solved"))
    payload = {
        "model": model_key,
        "done": done, "total": total,
        "pct": round(100 * done / total, 1) if total else 0.0,
        "rate_s_per_q": round(rate_s, 2),
        "eta_min": round(eta_min, 1),
        "solved": solved,
        "solve_rate_pct": round(100 * solved / done, 1) if done else 0.0,
        "status_counts": counts,
    }
    path.write_text(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--limit-per-topic", type=int, default=None)
    ap.add_argument("--topics", nargs="+", default=None, help="subset of topics")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]
    data_path = REPO / ds["path"] if not Path(ds["path"]).is_absolute() else Path(ds["path"])
    image_root = REPO / ds["image_root"] if not Path(ds["image_root"]).is_absolute() else Path(ds["image_root"])
    topics = set(args.topics or cfg["topics"])
    limit = args.limit_per_topic if args.limit_per_topic is not None else cfg["output"].get("limit_per_topic", 0)

    records = [json.loads(l) for l in open(data_path) if l.strip()]
    print(f"loaded {len(records)} records from {data_path}", flush=True)

    # Guard: steering prompts must not leak any option value.
    task_prompts.assert_no_leak(records)

    image_max_edge = int(cfg.get("attention", {}).get("image_max_edge", 768))
    work = build_worklist(records, image_root, image_max_edge, topics, limit)
    print(f"worklist: {len(work)} questions; per-topic "
          f"{dict(Counter(w['topic'] for w in work))}", flush=True)

    out_dir = (REPO / cfg["output"]["dir"] / cfg["model"]["key"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip questions already recorded (survives a SLURM timeout/requeue).
    done_qids, prior_counts = _load_done(out_dir)
    if done_qids:
        before = len(work)
        work = [w for w in work if w["record"].get("question_id") not in done_qids]
        print(f"resume: {len(done_qids)} already done, {len(work)}/{before} remaining", flush=True)

    col = Collector(cfg)
    solved_f = open(out_dir / "solved.jsonl", "a")     # append (resume-safe)
    unsolved_f = open(out_dir / "unsolved.jsonl", "a")
    progress_path = out_dir / "progress.json"

    t0 = time.time()
    status_counts = Counter(prior_counts)
    total = len(work) + len(done_qids)
    for n, item in enumerate(work, 1):
        res = col.process(item)
        status_counts[res["status"]] += 1
        f = unsolved_f if res["status"] == "unsolved" else solved_f
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()
        done = n + len(done_qids)
        if n % 5 == 0 or n == len(work):
            el = time.time() - t0
            rate = el / n
            eta_min = rate * (len(work) - n) / 60
            _write_progress(progress_path, done, total, dict(status_counts),
                            rate, eta_min, cfg["model"]["key"])
            print(f"  [{done}/{total}] {dict(status_counts)} "
                  f"({el/60:.1f} min, {rate:.1f}s/q, ETA {eta_min:.0f} min)", flush=True)
    col.close()
    solved_f.close()
    unsolved_f.close()

    meta = {"model": cfg["model"], "ladder": cfg["ladder"],
            "attention": cfg["attention"], "topics": sorted(topics),
            "dataset": {"path": str(data_path), "image_root": str(image_root)},
            "n_questions": len(work), "status_counts": dict(status_counts),
            "git_sha": _git_sha(), "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\ndone: {dict(status_counts)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
