"""lhrs_native_backend.py — native loader for LHRS-Bot-Nova (NJU-LHRS).

LHRS-Bot-Nova is a remote-sensing VLM: **Meta-Llama-3-8B-Instruct** + a
**SigLIP-so400m-patch14-384** vision encoder + a 272-query MoE perceiver
"bridge". It is satellite/aerial-only and single-image, shipped NOT in HF
`from_pretrained` layout but as raw `.pt` state dicts + a TextLoRA adapter, loaded
through the repo's own `lhrs` package (`build_model` + `custom_load_state_dict`).
So neither vLLM (custom perceiver, no converted ckpt) nor the HF AutoProcessor
backend can serve it — this is the suite's "native loader, batched" path, exactly
like llava_native is for GeoChat/SkySenseGPT.

What it does, per the repo's own cli_qa.py / main_vqa.py, with two deliberate
improvements over the reference single-sample loop:
  1. **batched** generate with left-padding + an attention_mask (cli_qa is
     single-sample; main_vqa supports batches but defaults to bs=1).
  2. **max_new_tokens=8** (the answer is one letter), not 512.

The prompt TEXT is the suite's shared MCQ prompt (prompt.SYSTEM_PROMPT folded
into the user turn + WorkItem.user_text) so LHRS-Bot is scored on byte-identical
question text to every other model — only the LLaMA-3 chat wrapper and the
`<image>` token placement are model-specific. The image is whatever the suite's
ablation produced (for our scope: the single marked-sat tile, red-dot marker
kept — a zero-shot transfer condition; LHRS-Bot never trained on the marker).

Load recipe (from Config/multi_modal_eval.yaml + UniBind.custom_load_state_dict):
  - build_model(config, activate_modal=("rgb","text")) builds the SigLIP tower,
    the MoE bridge, and the Llama-3 LLM (base pulled from the HF hub by name).
  - config.stage = 0 (eval) so the Stage3 TextLoRA is merge_and_unload()-ed.
  - custom_load_state_dict(Stage3/FINAL.pt) loads the RGB encoder + rgb_pooler
    from FINAL.pt and auto-loads the sibling Stage3/TextLoRA/ adapter.

Blackwell note: the repo hard-codes use_flash_attention_2=True for the Llama-3
LLM; flash_attn 2.6.1 has no sm_120 wheels, so text_modal.py is patched (vendored,
on the GPU node) to request attn_implementation="sdpa" instead — fast and correct
on Blackwell, no accuracy impact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from backends.base import Backend, ResultRow, WorkItem, resolve_model_id

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt

_DEFAULT_CKPT = "LHRS/LHRS-Bot-Nova"
# Eval config shipped in the repo (Config/multi_modal_eval.yaml). We read it for
# the SigLIP name / LLM path / pooler shape, then override the few control knobs.
_EVAL_YAML_REL = "Config/multi_modal_eval.yaml"


class LhrsNativeBackend(Backend):
    """Batched native inference for LHRS-Bot-Nova (384px SigLIP, 1 image)."""

    name = "lhrs_native"

    def __init__(self, model_cfg: dict, defaults: dict):
        if model_cfg.get("attention"):
            raise ValueError(
                "lhrs_native cannot expose attention — LHRS-Bot is loaded via the "
                "lhrs package with a custom MoE perceiver, not the eager HF path.")
        if model_cfg.get("lora_path"):
            raise ValueError(
                "lhrs_native serves the released Stage3 checkpoint (its TextLoRA is "
                "merged at load); it does not take an external PEFT adapter.")
        self.cfg = model_cfg
        self.defaults = defaults
        self.batch_size = int(model_cfg.get("batch_size", defaults.get("batch_size", 8)))
        self.max_new_tokens = int(
            model_cfg.get("max_new_tokens", defaults.get("max_new_tokens", 8)))
        # repo source dir (has the lhrs package + Config/). Required so we can read
        # the eval yaml and resolve the Stage3 checkpoint relative paths.
        self.repo_dir = Path(model_cfg["repo_dir"]).expanduser()
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        import ml_collections
        import yaml as _yaml
        from huggingface_hub import snapshot_download

        from lhrs.Dataset.build_transform import build_vlp_transform
        from lhrs.Dataset.conversation import default_conversation
        from lhrs.models import (
            DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, build_model,
            tokenizer_image_token)
        from lhrs.utils import type_dict

        self._tokenizer_image_token = tokenizer_image_token
        self._default_conversation = default_conversation
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN

        # --- assemble the config the repo's build_model expects -------------
        yaml_path = self.repo_dir / _EVAL_YAML_REL
        raw = _yaml.safe_load(open(yaml_path))
        config = ml_collections.config_dict.ConfigDict(raw)
        config.stage = 0          # eval: merge the TextLoRA at load
        config.accelerator = "gpu"
        config.tune_im_start = bool(config.get("tune_im_start", False))
        # The eval yaml points text.path at the GATED meta-llama/Meta-Llama-3-8B-
        # Instruct (the base LLM the perceiver+LoRA were trained on). If the HF
        # token lacks download grant, allow overriding it with a byte-identical
        # ungated MIRROR of the SAME weights (e.g. NousResearch/...). This is not
        # a model swap — identical arch/hidden/layers/vocab — just a re-host.
        llm_override = self.cfg.get("llm_path")
        if llm_override:
            print(f"  [lhrs_native] overriding base LLM path -> {llm_override} "
                  f"(was {config.text.path})", flush=True)
            config.text.path = llm_override

        # Resolve the Stage3 checkpoint (FINAL.pt). hf_id may be a hub repo
        # (download Stage3/*) or a local snapshot dir already containing Stage3/.
        hf_id = self.cfg.get("hf_id") or _DEFAULT_CKPT
        resolved = resolve_model_id(hf_id)
        if Path(resolved).exists() and (Path(resolved) / "Stage3" / "FINAL.pt").exists():
            stage3 = Path(resolved) / "Stage3" / "FINAL.pt"
        else:
            snap = snapshot_download(repo_id=hf_id, allow_patterns=["Stage3/*"])
            stage3 = Path(snap) / "Stage3" / "FINAL.pt"
        config.model_path = str(stage3)

        self.dtype = type_dict[config.dtype]   # float16 per the eval yaml
        print(f"  [lhrs_native] build_model (siglip={config.rgb_vision.vit_name}, "
              f"llm={config.text.path}, stage3={stage3.name}, bs={self.batch_size})",
              flush=True)

        self.model = build_model(config, activate_modal=("rgb", "text"))
        self.vision_processor = build_vlp_transform(config, is_train=False)
        self.model.to(self.dtype)
        self.model.custom_load_state_dict(config.model_path, strict=False)
        self.model.to("cuda")
        self.model.eval()
        self.config = config

        self.tokenizer = self.model.text.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"   # batched greedy: align answers
        self.device = torch.device("cuda")
        # Cache option-letter token ids for the prob_dict (first-token scores).
        self._letter_ids = {}
        for L in ("A", "B", "C", "D"):
            ids = self.tokenizer.encode(L, add_special_tokens=False)
            if ids:
                self._letter_ids[L] = ids[-1]

    # ------------------------------------------------------------- prompting
    def _build_prompt_str(self, item: WorkItem) -> str:
        """LLaMA-3 chat: system + user(<image>\\n + suite text) + assistant.

        Uses the suite's shared system+user text (identical wording to every
        other model). tune_im_start is False for Nova, so the image token is a
        bare `<image>\\n` prepended to the user turn (matching cli_qa.py).
        """
        conv = self._default_conversation.copy()
        n_img = len(item.images)
        img_tokens = (self.DEFAULT_IMAGE_TOKEN + "\n") * n_img
        user_body = f"{suite_prompt.SYSTEM_PROMPT}\n\n{item.user_text}"
        conv.append_message(conv.roles[0], img_tokens + user_body)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _preprocess_images(self, pil_list):
        """List of PIL -> [N,3,384,384] tensor (SigLIP processor), or None."""
        if not pil_list:
            return None
        pil_list = [im.convert("RGB") for im in pil_list]
        # vit arch path: HF SiglipImageProcessor via build_vlp_transform.
        px = self.vision_processor(pil_list, return_tensors="pt").pixel_values
        return px.to(self.device, dtype=self.dtype)

    # ----------------------------------------------------------------- run
    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        rows: list[ResultRow] = []
        n = len(items)
        for start in range(0, n, self.batch_size):
            batch = items[start:start + self.batch_size]
            rows.extend(self._run_batch(batch))
            if (start // self.batch_size) % 10 == 0:
                print(f"  [lhrs_native] {min(start + self.batch_size, n)}/{n}",
                      flush=True)
        return rows

    @torch.inference_mode()
    def _run_batch(self, batch: list[WorkItem]) -> list[ResultRow]:
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
            input_ids[i, max_len - t.size(0):] = t   # LEFT pad
            attn[i, max_len - t.size(0):] = 1
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        all_imgs = [im for it in batch for im in it.images]
        images = self._preprocess_images(all_imgs)

        out = self.model.generate(
            input_ids, images=images,
            attention_mask=attn,
            do_sample=False, temperature=1.0,
            max_new_tokens=self.max_new_tokens, use_cache=True,
            output_scores=True, return_dict_in_generate=True,
        )

        # When images are spliced via inputs_embeds, HF returns ONLY the generated
        # tokens in .sequences (the prompt isn't echoed). Decode them directly.
        seqs = out.sequences
        if seqs.shape[1] > self.max_new_tokens and seqs.shape[1] >= input_ids.shape[1]:
            seqs = seqs[:, input_ids.shape[1]:]   # prompt echoed -> strip it
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
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
