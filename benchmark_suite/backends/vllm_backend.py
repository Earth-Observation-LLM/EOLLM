"""vllm_backend.py — vLLM backend (raw / AWQ weights), batched, thinking OFF.

The fast path for raw and quantized models. vLLM schedules the whole worklist
across max_num_seqs in one llm.generate call. It CANNOT serve a LoRA adapter and
CANNOT expose attention (PagedAttention fuses the softmax) — run.py refuses both
combinations before reaching here.

Mirrors the proven engine in benchmark_modes_remote/run_modes_vllm.py: single
greedy decode, first-token logprobs -> normalized prob_dict over the option
letters. This backend only runs where vLLM and the model weights exist (the
lab-ws GPU node); locally it is never selected because `import vllm` fails.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from backends.base import Backend, ResultRow, WorkItem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt


def _extract_prob_dict(logprobs_at_0, valid_letters):
    prob = {l: 0.0 for l in valid_letters}
    if not logprobs_at_0:
        return prob
    for _, lp in logprobs_at_0.items():
        tok = lp.decoded_token.strip().upper()
        if tok in prob:
            prob[tok] = max(prob[tok], math.exp(lp.logprob))
    s = sum(prob.values())
    if s > 0:
        prob = {k: v / s for k, v in prob.items()}
    return prob


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_cfg: dict, defaults: dict):
        if model_cfg.get("attention"):
            raise ValueError("vLLM cannot expose attention — use the transformers backend.")
        if model_cfg.get("lora_path"):
            raise ValueError("vLLM cannot serve a LoRA adapter — use the transformers backend.")
        self.cfg = model_cfg
        self.defaults = defaults
        self.enable_thinking = model_cfg.get(
            "enable_thinking", defaults.get("enable_thinking", False))
        self.max_img_px = int(model_cfg.get("max_img_px", defaults.get("image_max_edge", 1120)))
        self._load()

    def _load(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        hf_id = self.cfg["hf_id"]
        print(f"  [vllm] loading {hf_id}", flush=True)
        self.llm = LLM(
            model=hf_id,
            dtype="bfloat16",
            max_model_len=int(self.cfg.get("max_model_len", self.defaults.get("max_model_len", 16384))),
            gpu_memory_utilization=float(self.cfg.get("gpu_mem_util", 0.90)),
            max_num_seqs=int(self.cfg.get("max_num_seqs", self.defaults.get("max_num_seqs", 64))),
            limit_mm_per_prompt={"image": 6},
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.sampling = SamplingParams(temperature=0, max_tokens=8, logprobs=20)

    def _chat_kwargs(self):
        return {} if self.enable_thinking else {"enable_thinking": False}

    def _build_prompt(self, item: WorkItem) -> str:
        conversation = [
            {"role": "system", "content": suite_prompt.SYSTEM_PROMPT},
            {"role": "user", "content":
                [{"type": "image"} for _ in item.images] + [{"type": "text", "text": item.user_text}]},
        ]
        try:
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False, **self._chat_kwargs())
        except TypeError:
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)

    def _resize(self, images):
        out = []
        for im in images:
            im = im.convert("RGB")
            if self.max_img_px:
                im.thumbnail((self.max_img_px, self.max_img_px))
            out.append(im)
        return out

    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        vinputs = []
        for it in items:
            pil = self._resize(it.images)
            prompt_text = self._build_prompt(it)
            vinputs.append({"prompt": prompt_text,
                            "multi_modal_data": {"image": pil} if pil else {}})
        print(f"  [vllm] generating {len(vinputs)} prompts (batched)...", flush=True)
        outputs = self.llm.generate(vinputs, sampling_params=self.sampling)

        rows = []
        for it, out in zip(items, outputs):
            res = out.outputs[0]
            raw = res.text.strip()
            options = it.record["options"]
            valid = sorted(options.keys())
            parsed = suite_parsing.parse_answer(raw, options=options)
            prob_dict = None
            if res.logprobs:
                prob_dict = _extract_prob_dict(res.logprobs[0], valid)
            rows.append(ResultRow(
                index=it.index, mode=it.mode,
                gold=suite_parsing.parse_letter(it.record.get("answer")),
                prediction=parsed.letter, raw_response=raw, prob_dict=prob_dict,
                hedged=parsed.hedged, refused=parsed.refused,
                n_images=len(it.images), roles=it.roles))
        return rows

    def close(self):
        pass
