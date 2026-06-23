"""llava_native_backend.py — native original-LLaVA loader for GeoChat & SkySenseGPT.

GeoChat and SkySenseGPT are LLaVA-1.5-7B forks (Vicuna-1.5 + CLIP-ViT-L/14
**interpolated to 504x504**), shipped in ORIGINAL-LLaVA format with a custom
`GeoChatLlamaForCausalLM`. They are NOT loadable by the HF Auto classes the other
backends use, have no HF chat template, and are single-image satellite-only. So
neither the vLLM backend (no original-LLaVA support, would need a converted
checkpoint) nor the HF backend (AutoProcessor / chat template) can serve them.

This is the suite's "Tier B" path from the integration brief: drive the model's
OWN `geochat` package loader (which correctly interpolates positional embeddings
to 504), with a BATCHED generate. Both models share the exact same code path —
only the checkpoint differs — so one backend serves both via `family:`.

What it does, per the GeoChat/SkySenseGPT eval scripts (batch_geochat_vqa.py),
with two deliberate improvements over their reference loop:
  1. **attention_mask** is built and passed (their script left-pads with id 0 and
     passes no mask, so the model attends to pad positions — a latent bug).
  2. **max_new_tokens=8** (the answer is one letter), not 256.

The TEXT of the prompt is the suite's shared MCQ prompt (prompt.SYSTEM_PROMPT +
WorkItem.user_text), so GeoChat/SkySenseGPT are scored on byte-identical question
text to every other model — only the `<image>` token placement and the Vicuna
USER:/ASSISTANT: wrapper are model-specific. The image itself is whatever the
suite's ablation produced for the item (for our scope: the single marked-sat
tile, with its red-dot marker kept — a zero-shot transfer condition).

Selection: run.py routes here for `family in {geochat, skysensegpt}` (or an
explicit `backend: llava_native`). The geochat package + a transformers==4.31 env
must be importable; that only exists in the dedicated `geochat` conda env on the
GPU node, so this backend is never selected locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from backends.base import Backend, ResultRow, WorkItem, resolve_model_id

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt

# Default HF checkpoints per family (overridable via model_cfg["hf_id"]).
_DEFAULT_CKPT = {
    "geochat": "MBZUAI/geochat-7B",
    "skysensegpt": "ll-13/SkySenseGPT-7B-clip-lora",
}


class LlavaNativeBackend(Backend):
    """Batched native-LLaVA inference for GeoChat / SkySenseGPT (504px, 1 image)."""

    name = "llava_native"

    def __init__(self, model_cfg: dict, defaults: dict):
        if model_cfg.get("attention"):
            raise ValueError(
                "llava_native cannot expose attention — these are original-LLaVA "
                "models loaded via the geochat package, not the eager HF path.")
        if model_cfg.get("lora_path"):
            raise ValueError(
                "llava_native serves a merged checkpoint, not a PEFT adapter.")
        self.cfg = model_cfg
        self.defaults = defaults
        self.family = model_cfg.get("family", "geochat")
        self.batch_size = int(model_cfg.get("batch_size", defaults.get("batch_size", 16)))
        self.max_new_tokens = int(
            model_cfg.get("max_new_tokens", defaults.get("max_new_tokens", 8)))
        self.image_res = int(model_cfg.get("image_res", 504))  # interpolated CLIP res
        self.conv_mode = model_cfg.get("conv_mode", "llava_v1")
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        from geochat.model.builder import load_pretrained_model
        from geochat.mm_utils import (
            tokenizer_image_token, get_model_name_from_path)
        from geochat.conversation import conv_templates
        from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        self._tokenizer_image_token = tokenizer_image_token
        self._conv_templates = conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN

        hf_id = self.cfg.get("hf_id") or _DEFAULT_CKPT[self.family]
        model_path = resolve_model_id(hf_id)
        model_name = get_model_name_from_path(model_path)
        # The geochat builder dispatches on substrings of model_name:
        #   - "geochat"  -> the LLaVA load path + vision-tower/image_processor setup
        #   - "lora"     -> the unmerged-adapter path (needs model_base)
        # SkySenseGPT's repo name lacks "geochat" (-> image_processor=None) but
        # CONTAINS "lora" despite shipping a MERGED checkpoint (no adapter files).
        # We normalize the name to a merged-geochat form: ensure "geochat" is
        # present and strip "lora" so the builder takes the plain merged path with
        # the CLIP tower wired up. Affects only branch selection, never weights.
        norm = model_name.lower()
        if "geochat" not in norm or "lora" in norm:
            model_name = "geochat-7b"
        print(f"  [llava_native] loading {model_path} (name={model_name}, "
              f"res={self.image_res}, conv={self.conv_mode}, bs={self.batch_size})",
              flush=True)
        # model_base=None: both ckpts are MERGED full models (SkySenseGPT's "lora"
        # in the repo name notwithstanding — it ships merged bin shards).
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name)
        self.model.eval()
        # Left padding so the answer (greedy first token) is at a fixed position
        # across the batch; LLaVA's reference loop left-pads but omits the mask.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        # Cache the option-letter token ids for the prob_dict (first-token scores).
        self._letter_ids = {}
        for L in ("A", "B", "C", "D"):
            ids = self.tokenizer.encode(L, add_special_tokens=False)
            if ids:
                self._letter_ids[L] = ids[-1]

    # ------------------------------------------------------------- prompting
    def _build_prompt_str(self, item: WorkItem) -> str:
        """Vicuna-1.5 prompt: system + USER: <image>\n<suite-text> ASSISTANT:.

        Uses the suite's shared system + user text (so question wording is
        identical to every other model). The leading <image> token(s) are what
        the geochat package splices vision patches into. These models are
        single-image; for our scope there is exactly one image, but we emit one
        <image> per provided image to stay honest if ever fed more.
        """
        conv = self._conv_templates[self.conv_mode].copy()
        n_img = len(item.images)
        img_tokens = (self.DEFAULT_IMAGE_TOKEN + "\n") * n_img
        # Fold the suite's system prompt into the user turn so the textual content
        # matches the other backends exactly (the conv template carries its own
        # generic Vicuna system string, which we keep as the chat scaffold).
        user_body = f"{suite_prompt.SYSTEM_PROMPT}\n\n{item.user_text}"
        conv.append_message(conv.roles[0], img_tokens + user_body)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _preprocess_images(self, pil_list):
        """List of PIL -> [N,3,res,res] tensor (or None for the no-image case).

        Mirrors GeoChat/SkySenseGPT: pass the PIL list straight to the CLIP
        processor with crop_size/size overridden to the interpolated resolution.
        """
        if not pil_list:
            return None
        pil_list = [im.convert("RGB") for im in pil_list]
        px = self.image_processor.preprocess(
            pil_list,
            crop_size={"height": self.image_res, "width": self.image_res},
            size={"shortest_edge": self.image_res},
            return_tensors="pt")["pixel_values"]
        return px.to(self.device, dtype=self.dtype)

    # ----------------------------------------------------------------- run
    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        rows: list[ResultRow] = []
        n = len(items)
        for start in range(0, n, self.batch_size):
            batch = items[start:start + self.batch_size]
            rows.extend(self._run_batch(batch))
            if (start // self.batch_size) % 10 == 0:
                print(f"  [llava_native] {min(start + self.batch_size, n)}/{n}",
                      flush=True)
        return rows

    @torch.inference_mode()
    def _run_batch(self, batch: list[WorkItem]) -> list[ResultRow]:
        # Tokenize each prompt with the image sentinel spliced in.
        id_list = [
            self._tokenizer_image_token(
                self._build_prompt_str(it), self.tokenizer,
                self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            for it in batch
        ]
        max_len = max(t.size(0) for t in id_list)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, t in enumerate(id_list):
            input_ids[i, max_len - t.size(0):] = t  # LEFT pad
            attn[i, max_len - t.size(0):] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        # All items in a batch share the same ablation mode and (for our scope)
        # exactly one image each; stack them into [B,3,res,res].
        all_imgs = [im for it in batch for im in it.images]
        images = self._preprocess_images(all_imgs)

        gen_kwargs = dict(
            attention_mask=attn,
            do_sample=False, num_beams=1,
            max_new_tokens=self.max_new_tokens, use_cache=True,
            output_scores=True, return_dict_in_generate=True,
        )
        if images is not None:
            gen_kwargs["images"] = images
        out = self.model.generate(input_ids, **gen_kwargs)

        seqs = out.sequences
        gen_only = seqs[:, input_ids.shape[1]:]
        texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        # First decoded-step logits -> normalized prob over the 4 option letters.
        first_logits = out.scores[0] if out.scores else None

        rows = []
        for i, it in enumerate(batch):
            raw = texts[i].strip()
            options = it.record["options"]
            valid = sorted(options.keys())
            parsed = suite_parsing.parse_answer(raw, options=options)
            prob_dict = None
            if first_logits is not None:
                prob_dict = self._prob_dict(first_logits[i], valid)
            rows.append(ResultRow(
                index=it.index, mode=it.mode,
                gold=suite_parsing.parse_letter(it.record.get("answer")),
                prediction=parsed.letter, raw_response=raw, prob_dict=prob_dict,
                hedged=parsed.hedged, refused=parsed.refused,
                n_images=len(it.images), roles=it.roles))
        return rows

    def _prob_dict(self, logits_row, valid_letters):
        """Softmax over the option-letter token ids at the first decode step."""
        ids = [self._letter_ids[L] for L in valid_letters if L in self._letter_ids]
        if not ids:
            return {L: 0.0 for L in valid_letters}
        sel = logits_row[ids]
        probs = torch.softmax(sel.float(), dim=-1).tolist()
        present = [L for L in valid_letters if L in self._letter_ids]
        return {L: float(p) for L, p in zip(present, probs)}

    def close(self):
        try:
            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass
