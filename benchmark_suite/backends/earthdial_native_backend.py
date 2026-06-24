"""earthdial_native_backend.py — native loader for EarthDial_4B_RGB (InternVL2).

EarthDial is a remote-sensing VLM built on the InternVL2-4B recipe:
**InternViT-300M** vision encoder + **Phi-3-Mini** LLM, loaded as
`earthdial.model.internvl_chat.InternVLChatModel` (trust_remote_code, bf16). It is
satellite/aerial-only and single-image (one tile, dynamically split into up to 6
448px sub-tiles + a thumbnail). vLLM can't serve the custom InternVL modules, and
the HF AutoProcessor path doesn't apply — so this is the suite's native-loader
path, like lhrs_native / llava_native.

What it does, per the repo's own inference.py / demo + the model's batch_chat:
  - InternVL dynamic high-resolution tiling: dynamic_preprocess(min_num=1,
    max_num=6, image_size=448, use_thumbnail=True), ImageNet normalization
    (matches config: dynamic_image_size, max_dynamic_patch=6, use_thumbnail).
  - InternVLChatModel.batch_chat(): builds the phi3-chat prompt per item, expands
    <image> into <img>+<IMG_CONTEXT>*num_image_token*num_patches+</img>, left-pads
    the batch, and runs a single batched generate. We pass max_new_tokens=8 (the
    answer is one letter).

The prompt TEXT is the suite's shared MCQ prompt (prompt.SYSTEM_PROMPT folded
into the user turn + WorkItem.user_text) so EarthDial is scored on byte-identical
question text to every other model. The image is the suite's ablation output (our
scope: the single marked-sat tile, red-dot marker kept — zero-shot transfer).

Blackwell note: the config sets vision_config.use_flash_attn=true, but the
InternViT attention class degrades gracefully to a naive eager path when flash_attn
isn't importable (verified in modeling_intern_vit.py), and the Phi-3 LLM doesn't
force FA2. So we run with NO flash_attn on sm_120 — we explicitly set
use_flash_attn=False on both sub-configs at load to avoid the import attempt.

NOTE on contamination (for the writeup, not the code): EarthDial's QA is OSM-
derived, so a high Family-1 score may reflect shared OSM priors with our OSM-
grounded labels, not pure visual skill. Flagged in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from backends.base import Backend, ResultRow, WorkItem, resolve_model_id

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parsing as suite_parsing
import prompt as suite_prompt

_DEFAULT_CKPT = "akshaydudhane/EarthDial_4B_RGB"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EarthDialNativeBackend(Backend):
    """Batched native inference for EarthDial_4B_RGB (448px InternVL tiling)."""

    name = "earthdial_native"

    def __init__(self, model_cfg: dict, defaults: dict):
        if model_cfg.get("attention"):
            raise ValueError(
                "earthdial_native cannot expose attention — InternVLChatModel is "
                "loaded via trust_remote_code, not the eager HF path.")
        if model_cfg.get("lora_path"):
            raise ValueError(
                "earthdial_native serves the merged EarthDial_4B checkpoint, not a "
                "PEFT adapter.")
        self.cfg = model_cfg
        self.defaults = defaults
        self.batch_size = int(model_cfg.get("batch_size", defaults.get("batch_size", 8)))
        self.max_new_tokens = int(
            model_cfg.get("max_new_tokens", defaults.get("max_new_tokens", 8)))
        self.image_size = int(model_cfg.get("image_size", 448))
        self.max_tiles = int(model_cfg.get("max_tiles", 6))   # config max_dynamic_patch
        self.use_thumbnail = bool(model_cfg.get("use_thumbnail", True))
        self.repo_dir = Path(model_cfg["repo_dir"]).expanduser()
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoTokenizer

        # repo's own InternVL chat model + the trained dynamic-tiling preprocess.
        sys.path.insert(0, str(self.repo_dir / "src"))
        from earthdial.model.internvl_chat import InternVLChatModel
        from earthdial.train.dataset import dynamic_preprocess

        self._dynamic_preprocess = dynamic_preprocess
        # Build the eval transform exactly like build_transform(is_train=False,
        # input_size=448, normalize_type='imagenet'): Resize -> ToTensor -> Norm.
        self._transform = T.Compose([
            T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
            T.Resize((self.image_size, self.image_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        hf_id = self.cfg.get("hf_id") or _DEFAULT_CKPT
        model_path = resolve_model_id(hf_id)
        print(f"  [earthdial_native] loading {model_path} "
              f"(tiles<= {self.max_tiles}, thumb={self.use_thumbnail}, "
              f"bs={self.batch_size}, no flash_attn)", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)
        self.model = InternVLChatModel.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            trust_remote_code=True).eval().cuda()
        # No flash_attn on Blackwell sm_120: force the eager InternViT path.
        try:
            self.model.config.vision_config.use_flash_attn = False
            if hasattr(self.model, "vision_model"):
                self.model.vision_model.encoder.use_flash_attn = False
        except Exception:
            pass

        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")
        # InternVL tokens.
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        img_ctx_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_ctx_id

    # --------------------------------------------------------- preprocessing
    def _tiles_for_image(self, pil):
        """One PIL -> [n_tiles,3,448,448] tensor via InternVL dynamic tiling."""
        pil = pil.convert("RGB")
        tiles = self._dynamic_preprocess(
            pil, min_num=1, max_num=self.max_tiles,
            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        px = torch.stack([self._transform(t) for t in tiles])
        return px.to(self.device, dtype=self.dtype)

    # ----------------------------------------------------------------- run
    def run(self, items: list[WorkItem]) -> list[ResultRow]:
        rows: list[ResultRow] = []
        n = len(items)
        for start in range(0, n, self.batch_size):
            batch = items[start:start + self.batch_size]
            rows.extend(self._run_batch(batch))
            if (start // self.batch_size) % 10 == 0:
                print(f"  [earthdial_native] {min(start + self.batch_size, n)}/{n}",
                      flush=True)
        return rows

    @torch.inference_mode()
    def _run_batch(self, batch: list[WorkItem]) -> list[ResultRow]:
        # For our scope each item has exactly one image; tile each independently
        # and record its tile count for batch_chat's num_patches_list.
        px_list, num_patches_list, questions = [], [], []
        for it in batch:
            # single-image satellite-only: take the first (only) image.
            pil = it.images[0]
            px = self._tiles_for_image(pil)
            px_list.append(px)
            num_patches_list.append(px.size(0))
            # The textual question = suite system + user text. batch_chat prepends
            # "<image>\n" itself when "<image>" not already present.
            questions.append(f"{suite_prompt.SYSTEM_PROMPT}\n\n{it.user_text}")

        pixel_values = torch.cat(px_list, dim=0)
        gen_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False,
                          num_beams=1)
        responses = self.model.batch_chat(
            self.tokenizer, pixel_values, questions, gen_config,
            num_patches_list=num_patches_list,
            IMG_CONTEXT_TOKEN=self.IMG_CONTEXT_TOKEN)

        rows = []
        for it, raw in zip(batch, responses):
            raw = (raw or "").strip()
            options = it.record["options"]
            parsed = suite_parsing.parse_answer(raw, options=options)
            rows.append(ResultRow(
                index=it.index, mode=it.mode,
                gold=suite_parsing.parse_letter(it.record.get("answer")),
                prediction=parsed.letter, raw_response=raw, prob_dict=None,
                hedged=parsed.hedged, refused=parsed.refused,
                n_images=len(it.images), roles=it.roles))
        return rows

    def close(self):
        try:
            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass
