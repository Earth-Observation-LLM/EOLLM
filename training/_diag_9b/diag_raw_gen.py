"""
diag_raw_gen.py — DECISIVE TEST for the 9B 2%-accuracy mystery.

Loads the local Qwen3.5-9B base + the pulled r16 seen_unseen adapter, runs greedy
generation on a sample of validation records, and DUMPS THE RAW DECODED STRING for
each under three configs:

  (1) thinking ON  (default template), max_new_tokens=16   <- what the callback used → 2%
  (2) thinking ON  (default template), max_new_tokens=128
  (3) thinking OFF (enable_thinking=False), max_new_tokens=16

If (1) shows '<think>...' truncated with no letter but (2)/(3) show a clean letter,
the 2% is an EVAL ARTIFACT (model is fine — fix eval, training continues).
If all three show empty/garbage/repetition, the model COLLAPSED (must retrain).

Run from repo root:
  cd ~/Development/EOLLM && conda run -n unsloth python training/_diag_9b/diag_raw_gen.py
"""
import os, sys, json, random
import glob
_snaps = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/"))
os.environ.setdefault("BASE_MODEL", _snaps[0] if _snaps else "Qwen/Qwen3.5-9B")
os.environ.setdefault("MODEL_FAMILY", "qwen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import SYSTEM_PROMPT
from data import convert_record
from eval_base import parse_letter, _pad_id

ADAPTER = os.path.join(os.path.dirname(__file__), "adapter")
SPLIT_DIR = "dataset_content/EODATA_compressed_final/splits_seen_unseen"
VAL = os.path.join(SPLIT_DIR, "validation.jsonl")
MAX_EDGE = 512
N = 24


def load_records():
    recs = [json.loads(l) for l in open(VAL)]
    # stratify: a few from each topic, ensure satellite_marked (5-image) included
    by_topic = {}
    for r in recs:
        by_topic.setdefault(r["topic"], []).append(r)
    rng = random.Random(123)
    picked = []
    for t, rs in sorted(by_topic.items()):
        picked.extend(rng.sample(rs, min(2, len(rs))))
    return picked[:N]


def build_inputs(rec, tokenizer, enable_thinking):
    converted = convert_record(rec, SPLIT_DIR, MAX_EDGE)
    user_msg = converted["messages"][1]
    images = [p["image"] for p in user_msg["content"] if p["type"] == "image"]
    inf_messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            p if p["type"] == "text" else {"type": "image"}
            for p in user_msg["content"]
        ]},
    ]
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(
        inf_messages, add_generation_prompt=True, tokenize=False, **kw
    )
    if not images:
        inputs = tokenizer(text=text, add_special_tokens=False, return_tensors="pt").to("cuda")
    else:
        img_input = images[0] if len(images) == 1 else images
        inputs = tokenizer(img_input, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    return inputs, len(images)


def gen(model, tokenizer, inputs, max_new):
    pad_id = _pad_id(tokenizer)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, use_cache=True,
                             do_sample=False, pad_token_id=pad_id)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=False).strip()


def main():
    from unsloth import FastVisionModel
    print(f"Loading base {os.environ['BASE_MODEL']} + adapter {ADAPTER} ...", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=os.environ["BASE_MODEL"],
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        use_gradient_checkpointing="unsloth", max_seq_length=8192,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    FastVisionModel.for_inference(model)
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is not None and hasattr(obj, "padding_side"):
            obj.padding_side = "left"
    print("Loaded.\n", flush=True)

    recs = load_records()
    configs = [
        ("think_ON__16  (callback setting)", None, 16),
        ("think_ON__128", None, 128),
        ("think_OFF_16", False, 16),
    ]
    score = {c[0]: 0 for c in configs}
    n = len(recs)

    for i, rec in enumerate(recs):
        gold = rec["answer"]
        print(f"\n{'='*70}\n[{i+1}/{n}] topic={rec['topic']}  mode={rec['image_mode']}  GOLD={gold}")
        for label, think, mnt in configs:
            inputs, nimg = build_inputs(rec, tokenizer, think)
            raw = gen(model, tokenizer, inputs, mnt)
            letter = parse_letter(raw.replace("<think>", " ").replace("</think>", " "))
            ok = letter == gold
            score[label] += ok
            shown = raw.replace("\n", "\\n")
            if len(shown) > 160:
                shown = shown[:160] + "…"
            print(f"  [{label:34s}] imgs={nimg} pred={letter} {'✓' if ok else '✗'} | raw={shown!r}")

    print(f"\n{'='*70}\nSCORES over {n} samples:")
    for label, _, _ in configs:
        print(f"  {label:34s}: {score[label]}/{n} = {score[label]/n:.1%}")


if __name__ == "__main__":
    main()
