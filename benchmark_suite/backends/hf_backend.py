"""hf_backend.py — transformers backend (base + LoRA), batched, greedy.

Handles any model loadable by unsloth.FastVisionModel, with or without a PEFT
LoRA adapter (which vLLM cannot serve). Produces, per item, the parsed answer
letter AND a normalized prob_dict over the record's option letters (from the
first generated token's logits) so the metrics layer can compute F1/ROC.

Proven techniques carried over from training/eval_base.py and
EzelinyumEvaluator/backend_unsloth.py:
  - system prompt as a plain STRING (Gemma template quirk), user = image+text parts
  - enable_thinking=False  (Qwen3.5 otherwise reasons past the answer -> 0% acc)
  - padding_side="left"    (batched decode: every row's answer starts at input_len)
  - batch grouped by image count to minimize padding waste
  - first-token logits -> letter prob_dict, summed over multi-spelling token ids
"""
from __future__ import annotations

from collections import defaultdict

import torch

from backends.base import Backend, ResultRow, WorkItem

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt


def _tok(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _letter_token_ids(processor, letters):
    tok = _tok(processor)
    out = {}
    for L in letters:
        ids = set()
        for spelling in (L, f" {L}", f"▁{L}"):
            try:
                enc = tok.encode(spelling, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            except Exception:
                pass
        out[L] = sorted(ids)
    return out


class HFBackend(Backend):
    name = "transformers"

    def __init__(self, model_cfg: dict, defaults: dict):
        self.cfg = model_cfg
        self.defaults = defaults
        self.max_new_tokens = model_cfg.get("max_new_tokens", defaults.get("max_new_tokens", 8))
        self.batch_size = int(model_cfg.get("batch_size", defaults.get("batch_size", 16)))
        self.enable_thinking = model_cfg.get(
            "enable_thinking", defaults.get("enable_thinking", False))
        self._load()

    def _load(self):
        from unsloth import FastVisionModel
        hf_id = str(Path(self.cfg["hf_id"]))
        # Resolve relative paths against the repo root.
        repo = Path(__file__).resolve().parent.parent.parent
        if not Path(hf_id).is_absolute():
            hf_id = str(repo / hf_id)
        print(f"  [hf] loading {hf_id}", flush=True)
        model, processor = FastVisionModel.from_pretrained(
            hf_id, load_in_4bit=False, load_in_8bit=False, dtype=torch.bfloat16,
        )
        lora = self.cfg.get("lora_path")
        if lora:
            lora_abs = lora if Path(lora).is_absolute() else str(repo / lora)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_abs)
            print(f"  [hf] adapter: {lora_abs}", flush=True)
        FastVisionModel.for_inference(model)
        tok = _tok(processor)
        if tok is not None:
            tok.padding_side = "left"
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
        self.model = model
        self.processor = processor
        self.pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

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

    @torch.no_grad()
    def _run_batch(self, items: list[WorkItem]) -> list[ResultRow]:
        texts = [self._apply_template(it) for it in items]
        images_list = [it.images for it in items]

        if all(len(im) == 0 for im in images_list):
            inputs = self.processor(
                text=texts, add_special_tokens=False, return_tensors="pt",
                padding=True, padding_side="left").to(self.model.device)
        else:
            inputs = self.processor(
                images_list, texts, add_special_tokens=False, return_tensors="pt",
                padding=True, padding_side="left").to(self.model.device)

        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            use_cache=True, return_dict_in_generate=True, output_scores=True,
            pad_token_id=self.pad_id)

        tok = _tok(self.processor)
        input_len = inputs["input_ids"].shape[1]
        rows = []
        for i, it in enumerate(items):
            gen_ids = out.sequences[i, input_len:]
            raw = tok.decode(gen_ids, skip_special_tokens=True).strip()
            valid = sorted(it.record["options"].keys())
            parsed = suite_parsing.parse_answer(raw, options=it.record["options"])
            prob_dict = None
            if out.scores:
                probs = torch.softmax(out.scores[0][i].float(), dim=-1)
                lid = _letter_token_ids(self.processor, valid)
                pd = {L: float(sum(probs[t].item() for t in ids if t < probs.shape[0]))
                      for L, ids in lid.items()}
                s = sum(pd.values())
                if s > 0:
                    pd = {L: v / s for L, v in pd.items()}
                prob_dict = pd
            rows.append(ResultRow(
                index=it.index, mode=it.mode,
                gold=suite_parsing.parse_letter(it.record.get("answer")),
                prediction=parsed.letter, raw_response=raw, prob_dict=prob_dict,
                hedged=parsed.hedged, refused=parsed.refused,
                n_images=len(it.images), roles=it.roles))
        return rows

    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        # Group by image count so batches pad uniformly; preserve original order
        # on output via the stored position.
        order = sorted(range(len(items)), key=lambda k: len(items[k].images))
        results: list[ResultRow | None] = [None] * len(items)
        done = 0
        i = 0
        while i < len(order):
            chunk_idx = order[i:i + self.batch_size]
            batch = [items[k] for k in chunk_idx]
            try:
                rows = self._run_batch(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows = []
                for it in batch:  # one-at-a-time fallback on OOM
                    rows.extend(self._run_batch([it]))
            for k, r in zip(chunk_idx, rows):
                results[k] = r
            done += len(batch)
            print(f"    [hf] {done}/{len(items)}", flush=True)
            i += self.batch_size
        return results  # type: ignore

    def close(self):
        import gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
