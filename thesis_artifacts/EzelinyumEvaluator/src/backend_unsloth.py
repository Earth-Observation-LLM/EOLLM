"""Unsloth backend — Qwen3.5-4B + optional LoRA, with BATCHED generation."""
from __future__ import annotations
import torch
from typing import Optional
from PIL import Image


def load_unsloth_model(
    hf_id: str,
    lora_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
):
    from unsloth import FastVisionModel
    from peft import PeftModel

    model, processor = FastVisionModel.from_pretrained(
        hf_id,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=dtype,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    FastVisionModel.for_inference(model)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tok is not None:
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
    return model, processor


def _build_messages(prompt: str, images: list[Image.Image]) -> list[dict]:
    content = []
    for im in images:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _letter_token_ids(processor, letters: list[str]) -> dict[str, list[int]]:
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
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


@torch.no_grad()
def generate_batch_unsloth(
    model,
    processor,
    batch: list[dict],   # list of {"images": [...], "prompt": str, "valid_letters": [str]}
    max_new_tokens: int = 8,
) -> list[tuple[str, dict[str, float]]]:
    """Batched greedy generation; returns list of (raw, prob_dict) aligned to batch order."""
    if not batch:
        return []

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Build per-example chat templates first.
    # enable_thinking=False forces an empty <think></think> block so the model
    # jumps straight to the answer instead of generating reasoning tokens.
    texts = []
    all_images: list[list[Image.Image]] = []
    for ex in batch:
        msgs = _build_messages(ex["prompt"], ex["images"])
        text = processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        texts.append(text)
        all_images.append(ex["images"])

    # Some Qwen-VL processors accept a flat list of images and a parallel
    # list of texts; others require nested. The unsloth processor expects
    # a flat list with image_grid info derived from text image_pad tokens.
    flat_images = [im for sublist in all_images for im in sublist]

    try:
        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        # Fallback: process each example individually with the same backend
        # (NOT via generate_one_unsloth which would recurse).
        results = []
        for ex in batch:
            try:
                single_text = texts[batch.index(ex)] if ex in batch else None
                single_inputs = processor(
                    text=[single_text] if single_text else None,
                    images=ex["images"] if ex["images"] else None,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                single_out = model.generate(
                    **single_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=tok.pad_token_id,
                )
                il = single_inputs["input_ids"].shape[1]
                raw = tok.decode(single_out.sequences[0, il:], skip_special_tokens=True)
                pd: dict[str, float] = {}
                if single_out.scores:
                    probs = torch.softmax(single_out.scores[0][0].float(), dim=-1)
                    letter_ids = _letter_token_ids(processor, ex["valid_letters"])
                    for L, ids in letter_ids.items():
                        pd[L] = float(sum(probs[t].item() for t in ids if t < probs.shape[0]))
                    s = sum(pd.values())
                    if s > 0: pd = {L: v / s for L, v in pd.items()}
                results.append((raw, pd))
            except Exception as e2:
                results.append((f"<error: {e2}>", {}))
        return results

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.pad_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    seqs = out.sequences  # [B, input_len + gen]
    scores = out.scores  # tuple of [B, vocab] per step

    raws: list[str] = []
    prob_dicts: list[dict[str, float]] = []
    for i, ex in enumerate(batch):
        gen_ids = seqs[i, input_len:]
        raw = tok.decode(gen_ids, skip_special_tokens=True)
        prob_dict: dict[str, float] = {}
        if scores:
            first_logits = scores[0][i]
            probs = torch.softmax(first_logits.float(), dim=-1)
            letter_ids = _letter_token_ids(processor, ex["valid_letters"])
            for L, ids in letter_ids.items():
                prob_dict[L] = float(sum(probs[t].item() for t in ids if t < probs.shape[0]))
            s = sum(prob_dict.values())
            if s > 0:
                prob_dict = {L: v / s for L, v in prob_dict.items()}
        raws.append(raw)
        prob_dicts.append(prob_dict)
    return list(zip(raws, prob_dicts))


@torch.no_grad()
def generate_one_unsloth(
    model,
    processor,
    images: list[Image.Image],
    prompt: str,
    valid_letters: list[str],
    max_new_tokens: int = 8,
) -> tuple[str, dict[str, float]]:
    """Single-record fallback (used when batched call fails for a model)."""
    result = generate_batch_unsloth(
        model, processor,
        [{"images": images, "prompt": prompt, "valid_letters": valid_letters}],
        max_new_tokens=max_new_tokens,
    )
    return result[0] if result else ("", {})
