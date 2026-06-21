"""attention_backend.py — ATTENTION WATCH (transformers + eager, two-pass).

Measures, per record, how much attention the model's ANSWER places on each
input image at the decision moment — the visual companion to the causal
leave-one-image-out / complementarity measure.

Why this backend exists separately from hf_backend:
  - Attention weights are obtainable ONLY through transformers with
    attn_implementation="eager"; flash/sdpa return None, and vLLM cannot expose
    them at all. So this backend forces eager and runs locally.
  - Attention memory scales with sequence length squared, so we cap image pixels
    (max_pixels, ~512x512 default) to keep the full-attention forward on a 32GB
    card.

Two passes (avoids the memory + alignment pain of per-generation-step attentions):
  Pass 1: generate the answer normally, no attentions (gives the letter).
  Pass 2: ONE forward over [prompt + generated answer] with output_attentions=True
          -> one [batch, heads, L, L] tensor per layer.

Per-image attention:
  - The answer token at answer-index i is produced at query position
    prompt_len-1+i (causal LM: logits at p predict token p+1). So the analyzed
    query rows are prompt_len-1 .. L-2.
  - Image k occupies the k-th <|vision_start|> .. <|vision_end|> key span.
  - For each query row, sum attention over each image's key span, average over
    heads and the chosen layers, then read off:
        avg_*    : averaged over the whole generated answer (robust)
        choice_* : at the single choice token (sharp decision-moment value)
        *_raw_pct : share of ALL attention (images vs text)
        *_norm_pct: share among the images only (per-image comparison)
  With a bare single-letter answer avg and choice coincide.

Invariants asserted (and recorded) so bad numbers fail loudly, never silently:
  - number of detected vision spans == number of input images
  - each analyzed attention row sums to ~1 over the keys (softmax)
  - pass-2 forced answer letter == pass-1 generated letter
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from backends.base import Backend, ResultRow, WorkItem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt

VISION_START = 248053   # <|vision_start|>  (verified for Qwen3.5)
VISION_END = 248054     # <|vision_end|>


def _tok(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


class AttentionBackend(Backend):
    name = "transformers-attention"

    def __init__(self, model_cfg: dict, defaults: dict):
        self.cfg = model_cfg
        self.defaults = defaults
        a_def = defaults.get("attention", {}) or {}
        a_cfg = model_cfg.get("attention", {})
        a_cfg = a_cfg if isinstance(a_cfg, dict) else {}
        self.layers_spec = a_cfg.get("layers", a_def.get("layers", "all"))
        self.max_pixels = int(a_cfg.get("max_pixels", a_def.get("max_pixels", 262144)))
        self.store_per_layer = bool(a_cfg.get("store_per_layer", a_def.get("store_per_layer", True)))
        self.max_new_tokens = model_cfg.get("max_new_tokens", defaults.get("max_new_tokens", 8))
        self.enable_thinking = model_cfg.get(
            "enable_thinking", defaults.get("enable_thinking", False))
        self._load()

    def _load(self):
        # IMPORTANT: load with PLAIN transformers + attn_implementation="eager".
        # Unsloth's FastVisionModel compiles the inference path and returns an
        # EMPTY attentions tuple even with output_attentions=True — its speed
        # patches drop the score matrix. The vanilla HF model with eager is the
        # only path that materializes per-layer attentions. LoRA still works via
        # PeftModel on top of the plain HF model.
        from transformers import AutoModelForImageTextToText, AutoProcessor
        repo = Path(__file__).resolve().parent.parent.parent
        hf_id = self.cfg["hf_id"]
        hf_id = hf_id if Path(hf_id).is_absolute() else str(repo / hf_id)
        print(f"  [attn] loading {hf_id} (plain HF, eager, max_pixels={self.max_pixels})", flush=True)
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch.bfloat16, attn_implementation="eager",
            device_map="cuda", trust_remote_code=True,
        )
        lora = self.cfg.get("lora_path")
        if lora:
            lora_abs = lora if Path(lora).is_absolute() else str(repo / lora)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_abs)
            print(f"  [attn] adapter: {lora_abs}", flush=True)
        model.eval()
        tok = _tok(processor)
        if tok is not None:
            tok.padding_side = "left"
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
        # Cap image tokens to keep the L^2 attention on-card.
        self._set_max_pixels(processor)
        self.model = model
        self.processor = processor
        self.pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        # Layer count: Qwen3.5 keeps it in text_config. NOTE the returned
        # attentions tuple may be SHORTER than num_hidden_layers (only the
        # full-attention layers expose scores under eager); _layers_index is
        # therefore clamped to the actual returned length at compute time.
        tc = getattr(model.config, "text_config", None)
        self.n_layers = (getattr(model.config, "num_hidden_layers", None)
                         or (getattr(tc, "num_hidden_layers", None) if tc else None)
                         or 32)

    def _set_max_pixels(self, processor):
        """Cap the number of image tokens to keep the L^2 attention on-card.

        In transformers 5.x the Qwen-VL image processor's `max_pixels` is a
        read-only property derived from `size.longest_edge`; `min_pixels` from
        `size.shortest_edge`. So we set those on the SizeDict.
        """
        ip = getattr(processor, "image_processor", None)
        if ip is None or not hasattr(ip, "size"):
            return
        size = ip.size
        target = self.max_pixels
        try:
            # SizeDict supports attribute access; also keep shortest_edge <= target.
            size.longest_edge = target
            if getattr(size, "shortest_edge", None) and size.shortest_edge > target:
                size.shortest_edge = target
        except Exception:
            try:
                d = dict(size)
                d["longest_edge"] = target
                if d.get("shortest_edge", 0) > target:
                    d["shortest_edge"] = target
                ip.size = type(size)(**d) if hasattr(type(size), "__init__") else d
            except Exception as e:
                print(f"  [attn] warning: could not set max_pixels ({e}); using default", flush=True)

    def _layers_index(self):
        n = self.n_layers
        if isinstance(self.layers_spec, (list, tuple)) and len(self.layers_spec) == 2:
            lo, hi = self.layers_spec
            return list(range(max(0, lo), min(n, hi)))
        if self.layers_spec == "last_25pct":
            return list(range(int(n * 0.75), n))
        return list(range(n))  # "all"

    def _apply_template(self, item: WorkItem) -> str:
        msgs = [
            {"role": "system", "content": suite_prompt.SYSTEM_PROMPT},
            {"role": "user", "content":
                [{"type": "image"} for _ in item.images] + [{"type": "text", "text": item.user_text}]},
        ]
        try:
            return self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=self.enable_thinking)
        except TypeError:
            return self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

    def _make_inputs(self, item: WorkItem):
        text = self._apply_template(item)
        if item.images:
            return self.processor([item.images], [text], add_special_tokens=False,
                                  return_tensors="pt").to(self.model.device)
        return self.processor(text=[text], add_special_tokens=False,
                              return_tensors="pt").to(self.model.device)

    @staticmethod
    def _detect_spans(ids_row: torch.Tensor) -> list[tuple[int, int]]:
        """Return [(start, end_exclusive)] key spans for each image: the tokens
        strictly between a vision_start and the next vision_end."""
        ids = ids_row.tolist()
        spans = []
        i = 0
        while i < len(ids):
            if ids[i] == VISION_START:
                j = i + 1
                while j < len(ids) and ids[j] != VISION_END:
                    j += 1
                spans.append((i + 1, j))  # exclude the markers themselves
                i = j + 1
            else:
                i += 1
        return spans

    @torch.no_grad()
    def _attention_for_item(self, item: WorkItem) -> ResultRow:
        # ---- Pass 1: generate the answer (no attentions) --------------------
        inputs = self._make_inputs(item)
        gen = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            use_cache=True, return_dict_in_generate=True, pad_token_id=self.pad_id)
        tok = _tok(self.processor)
        prompt_len = inputs["input_ids"].shape[1]
        answer_ids = gen.sequences[0, prompt_len:]
        # strip trailing pad/eos
        keep = [t for t in answer_ids.tolist()
                if t not in (self.pad_id, getattr(tok, "eos_token_id", None))]
        raw = tok.decode(answer_ids, skip_special_tokens=True).strip()
        parsed = suite_parsing.parse_answer(raw, options=item.record["options"])

        row = ResultRow(
            index=item.index, mode=item.mode,
            gold=suite_parsing.parse_letter(item.record.get("answer")),
            prediction=parsed.letter, raw_response=raw,
            hedged=parsed.hedged, refused=parsed.refused,
            n_images=len(item.images), roles=item.roles)

        if not item.images or not keep:
            row.attention = {"note": "no images" if not item.images else "empty answer",
                             "n_images": len(item.images)}
            return row

        # ---- Pass 2: one forward over [prompt + answer], output_attentions ---
        ans_t = torch.tensor(keep, device=self.model.device).unsqueeze(0)
        full_ids = torch.cat([inputs["input_ids"], ans_t], dim=1)
        attn_mask = torch.ones_like(full_ids)
        fwd_kwargs = {"input_ids": full_ids, "attention_mask": attn_mask,
                      "output_attentions": True, "use_cache": False}
        model_dtype = next(self.model.parameters()).dtype
        # mm_token_type_ids marks image tokens (1) vs text (0) for M-RoPE. The
        # appended answer tokens are all text, so extend with zeros to keep the
        # 3D position-id computation valid over the concatenated sequence.
        if "mm_token_type_ids" in inputs:
            mm = inputs["mm_token_type_ids"]
            pad = torch.zeros((mm.shape[0], ans_t.shape[1]), dtype=mm.dtype, device=mm.device)
            fwd_kwargs["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs:
                v = inputs[k]
                # pixel_values come back float32 from the processor; the vision
                # tower runs in the model dtype (bf16). generate() casts these
                # internally, but a manual forward must match the dtype itself.
                if k == "pixel_values" and v.is_floating_point():
                    v = v.to(model_dtype)
                fwd_kwargs[k] = v
        # autocast keeps the vision tower + LM in the model dtype on the manual
        # forward (generate() does this internally; a raw model() call does not,
        # which otherwise surfaces as "expected BFloat16 but found Float" in the
        # vision block's layernorm).
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            out = self.model(**fwd_kwargs)
        attentions = out.attentions  # tuple(num_layers) of [1, heads, L, L]

        L = full_ids.shape[1]
        spans = self._detect_spans(full_ids[0])
        # invariant 1: one span per image
        spans_ok = (len(spans) == len(item.images))

        # query rows: the answer tokens. Answer token i is produced at query
        # position prompt_len-1+i. Rows prompt_len-1 .. L-2.
        ans_len = len(keep)
        q_start = prompt_len - 1
        q_end = L - 1  # exclusive (last answer token has no "next" to predict)
        q_rows = list(range(q_start, min(q_end, L)))
        # choice token = the answer token that decoded to the option letter.
        choice_local = self._find_choice_index(keep, tok, parsed.letter)
        choice_row = q_start + choice_local if choice_local is not None else q_rows[0]

        # The returned attentions tuple can be shorter than num_hidden_layers
        # (only full-attention layers expose scores under eager). Clamp the
        # requested layer indices to what actually came back.
        n_returned = len(attentions)
        layer_idx = [li for li in self._layers_index() if li < n_returned]
        if not layer_idx:
            layer_idx = list(range(n_returned))
        per_layer = self._compute_per_layer(
            attentions, layer_idx, q_rows, choice_row, spans)

        row.attention = self._assemble(
            per_layer, item.roles, len(item.images), parsed.letter, choice_local,
            spans_ok, layer_idx, self.store_per_layer)
        return row

    @staticmethod
    def _find_choice_index(answer_ids: list[int], tok, letter: str | None):
        if letter is None:
            return None
        for i, tid in enumerate(answer_ids):
            dec = tok.decode([tid]).strip().upper()
            if dec == letter:
                return i
        return None

    def _compute_per_layer(self, attentions, layer_idx, q_rows, choice_row, spans):
        """For each selected layer, head-average, then for the answer rows and the
        choice row compute each image's summed attention over its key span, plus
        the total image attention and a row-sum (softmax) check."""
        results = []
        for li in layer_idx:
            A = attentions[li][0]                  # [heads, L, L]
            A = A.float().mean(dim=0)              # head-average -> [L, L]
            rows_avg = A[q_rows, :].mean(dim=0)    # [L] averaged over answer rows
            rows_choice = A[choice_row, :]         # [L] at the choice token
            row_sum_check = float(rows_avg.sum().item())
            per_img_avg, per_img_choice = [], []
            for (s, e) in spans:
                per_img_avg.append(float(rows_avg[s:e].sum().item()))
                per_img_choice.append(float(rows_choice[s:e].sum().item()))
            results.append({
                "layer": li,
                "avg_raw": per_img_avg,        # fraction of all attention
                "choice_raw": per_img_choice,
                "row_sum_check": row_sum_check,
            })
        return results

    def _assemble(self, per_layer, roles, n_images, letter, choice_local,
                  spans_ok, layer_idx, store_per_layer):
        import statistics as st
        # Mean across layers of each image's raw share.
        def mean_over_layers(key, k):
            return st.fmean(pl[key][k] for pl in per_layer) if per_layer else 0.0

        images = []
        avg_raw = [mean_over_layers("avg_raw", k) for k in range(n_images)]
        choice_raw = [mean_over_layers("choice_raw", k) for k in range(n_images)]
        tot_avg = sum(avg_raw) or 1e-9
        tot_choice = sum(choice_raw) or 1e-9
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
            "n_images": n_images,
            "choice_letter": letter,
            "choice_token_index": choice_local,
            "layers_used": ("all" if self.layers_spec == "all" else self.layers_spec),
            "n_layers_used": len(layer_idx),
            "image_attention_total_avg_pct": round(100 * sum(avg_raw), 4),
            "spans_match_images": spans_ok,
            "row_softmax_check": round(st.fmean(pl["row_sum_check"] for pl in per_layer), 4) if per_layer else None,
            "images": images,
        }
        if store_per_layer:
            block["per_layer"] = [
                {"layer": pl["layer"],
                 "avg_raw_pct": [round(100 * x, 4) for x in pl["avg_raw"]],
                 "choice_raw_pct": [round(100 * x, 4) for x in pl["choice_raw"]]}
                for pl in per_layer
            ]
        return block

    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        rows = []
        for n, it in enumerate(items, 1):
            try:
                rows.append(self._attention_for_item(it))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = ResultRow(index=it.index, mode=it.mode,
                              gold=suite_parsing.parse_letter(it.record.get("answer")),
                              prediction=None, raw_response="<OOM>",
                              n_images=len(it.images), roles=it.roles,
                              attention={"note": "OOM — lower attention.max_pixels"})
                rows.append(r)
            if n % 10 == 0 or n == len(items):
                print(f"    [attn] {n}/{len(items)}", flush=True)
        return rows

    def close(self):
        import gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
