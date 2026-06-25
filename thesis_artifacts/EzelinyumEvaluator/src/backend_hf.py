"""HF transformers backend — baselines with BATCHED generation."""
from __future__ import annotations
import torch
from typing import Optional
from PIL import Image


def load_hf_model(hf_id: str, dtype: torch.dtype = torch.bfloat16):
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        AutoModelForImageTextToText,
    )

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else None
    if tok is not None:
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

    last_err = None
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        try:
            model = cls.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            return model, processor
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load {hf_id}: {last_err}")


def _build_messages(prompt: str, n_images: int) -> list[dict]:
    content = []
    for _ in range(n_images):
        content.append({"type": "image"})
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
        if not ids:
            for spelling in (L, f" {L}"):
                try:
                    enc = tok.encode(spelling, add_special_tokens=False)
                    if enc:
                        ids.add(enc[0])
                except Exception:
                    pass
        out[L] = sorted(ids)
    return out


@torch.no_grad()
def generate_batch_hf(
    model,
    processor,
    batch: list[dict],
    max_new_tokens: int = 8,
) -> list[tuple[str, dict[str, float]]]:
    if not batch:
        return []

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    texts = []
    all_images: list[list[Image.Image]] = []
    for ex in batch:
        msgs = _build_messages(ex["prompt"], len(ex["images"]))
        if hasattr(processor, "apply_chat_template"):
            try:
                text = processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = ex["prompt"]
        else:
            text = ex["prompt"]
        texts.append(text)
        all_images.append(ex["images"])

    flat_images = [im for sub in all_images for im in sub]

    def _proc_one(ex, text):
        if ex["images"]:
            return processor(
                text=[text],
                images=ex["images"],
                padding=True,
                return_tensors="pt",
            )
        return processor(text=[text], padding=True, return_tensors="pt")

    try:
        if flat_images:
            inputs = processor(
                text=texts,
                images=flat_images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=texts, padding=True, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if tok is not None and tok.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tok.pad_token_id

        out = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        results = []
        for i, ex in enumerate(batch):
            gen_ids = out.sequences[i, input_len:]
            raw = processor.decode(gen_ids, skip_special_tokens=True) if hasattr(processor, "decode") \
                else tok.decode(gen_ids, skip_special_tokens=True)
            pd: dict[str, float] = {}
            if out.scores:
                probs = torch.softmax(out.scores[0][i].float(), dim=-1)
                letter_ids = _letter_token_ids(processor, ex["valid_letters"])
                for L, ids in letter_ids.items():
                    pd[L] = float(sum(probs[t].item() for t in ids if t < probs.shape[0]))
                s = sum(pd.values())
                if s > 0:
                    pd = {L: v / s for L, v in pd.items()}
            results.append((raw, pd))
        return results
    except Exception:
        # Fallback to one-at-a-time.
        results = []
        for i, ex in enumerate(batch):
            try:
                inp = _proc_one(ex, texts[i])
                inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                if tok is not None and tok.pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = tok.pad_token_id
                out = model.generate(**inp, **gen_kwargs)
                il = inp["input_ids"].shape[1] if "input_ids" in inp else 0
                raw = processor.decode(out.sequences[0, il:], skip_special_tokens=True) if hasattr(processor, "decode") \
                    else tok.decode(out.sequences[0, il:], skip_special_tokens=True)
                pd: dict[str, float] = {}
                if out.scores:
                    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
                    letter_ids = _letter_token_ids(processor, ex["valid_letters"])
                    for L, ids in letter_ids.items():
                        pd[L] = float(sum(probs[t].item() for t in ids if t < probs.shape[0]))
                    s = sum(pd.values())
                    if s > 0: pd = {L: v / s for L, v in pd.items()}
                results.append((raw, pd))
            except Exception as e2:
                results.append((f"<error: {e2}>", {}))
        return results


# Compat shim (kept for now in case anything imports it).
@torch.no_grad()
def generate_one_hf(model, processor, images, prompt, valid_letters, max_new_tokens=8):
    r = generate_batch_hf(model, processor,
        [{"images": images, "prompt": prompt, "valid_letters": valid_letters}],
        max_new_tokens=max_new_tokens)
    return r[0] if r else ("", {})
