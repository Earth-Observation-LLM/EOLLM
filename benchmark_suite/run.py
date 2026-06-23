#!/usr/bin/env python3
"""run.py — the benchmark_suite entry point.

Reads config.yaml, and for each configured model:
  1. resolves a backend from (attention, lora_path, availability),
  2. builds a worklist of (record x applicable ablation mode), constructing the
     post-ablation images (images.build_images + modes.filter) and the prompt
     once per item,
  3. runs the backend,
  4. writes per-mode predictions.jsonl + report.json (+ attention.jsonl when on)
     and a meta.json with full provenance.

Usage:
    conda activate unsloth
    python run.py                          # all models in config.yaml
    python run.py --models qwen35_4b_base  # subset
    python run.py --limit-per-topic 3      # smoke
    python run.py --config my.yaml
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

import yaml

SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE))

import images as suite_images
import modes as suite_modes
import prompt as suite_prompt
import parsing as suite_parsing
import metrics as suite_metrics
from backends.base import WorkItem


# ---------------------------------------------------------------------------
# Backend resolution — the all-rounder rule
# ---------------------------------------------------------------------------

def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


# Original-LLaVA families (GeoChat / SkySenseGPT): 504px, single-image,
# satellite-only forks served by the geochat package. They have no HF chat
# template and aren't vLLM/AutoProcessor loadable, so they always route to the
# native LLaVA backend regardless of vLLM availability.
LLAVA_NATIVE_FAMILIES = {"geochat", "skysensegpt"}


def resolve_backend(model_cfg: dict) -> str:
    """Decide which backend to use for a model, honoring an explicit override.

    Raises on the one impossible combination: vLLM + attention.
    """
    attention = bool(model_cfg.get("attention", False))
    lora = bool(model_cfg.get("lora_path"))
    forced = model_cfg.get("backend")
    family = model_cfg.get("family")

    if forced:
        if forced == "vllm" and family in LLAVA_NATIVE_FAMILIES:
            raise ValueError(
                f"[{model_cfg['key']}] family {family!r} is original-LLaVA format "
                f"(504px, custom GeoChatLlamaForCausalLM) — vLLM cannot load it. "
                f"Use backend: llava_native (the default for this family).")
        if forced == "vllm" and attention:
            raise ValueError(
                f"[{model_cfg['key']}] backend: vllm with attention: true is "
                f"impossible — vLLM fuses the attention softmax and cannot expose "
                f"weights. Use backend: transformers for attention."
            )
        if forced == "vllm" and lora:
            raise ValueError(
                f"[{model_cfg['key']}] backend: vllm with a lora_path is "
                f"unsupported — vLLM cannot serve a PEFT adapter. Use transformers."
            )
        return forced

    if family in LLAVA_NATIVE_FAMILIES:
        return "llava_native"      # 504px original-LLaVA; geochat package loader
    if attention:
        return "transformers"      # only path that can capture attention
    if lora:
        return "transformers"      # vLLM can't serve LoRA
    return "vllm" if _vllm_available() else "transformers"


# ---------------------------------------------------------------------------
# Worklist construction
# ---------------------------------------------------------------------------

def load_records(data_path: Path, limit_per_topic: int) -> list[dict]:
    recs = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    if limit_per_topic:
        by_t = defaultdict(list)
        for r in recs:
            by_t[r.get("topic")].append(r)
        recs = [r for rs in by_t.values() for r in rs[:limit_per_topic]]
    return recs


def build_worklist(records, image_root, requested_modes, image_max_edge,
                   only_topics=None):
    """One WorkItem per (record, applicable mode). Images built once per item.

    A record whose images fail to build (missing files etc.) is skipped with a
    warning rather than crashing the run. `only_topics` (a set) restricts the run
    to those topics — used to confine satellite-only models (GeoChat /
    SkySenseGPT) to the 9 Family-1 overhead topics.
    """
    work: list[WorkItem] = []
    skipped = 0
    for idx, rec in enumerate(records):
        topic = rec.get("topic", "")
        if only_topics is not None and topic not in only_topics:
            continue
        rec_modes = suite_modes.resolve_modes(topic, requested_modes)
        try:
            full_imgs = suite_images.build_images(rec, image_root, image_max_edge)
        except Exception as e:  # missing image / bad record — skip, don't die
            skipped += 1
            if skipped <= 10:
                print(f"  [skip] {rec.get('question_id')}: {type(e).__name__}: {e}", flush=True)
            continue
        for mode in rec_modes:
            sel = suite_modes.filter_images_for_mode(full_imgs, mode)
            roles = [s["role"] for s in sel]
            pil = [s["image"] for s in sel]
            text = suite_prompt.build_user_text(rec, roles)
            work.append(WorkItem(index=idx, record=rec, mode=mode,
                                 images=pil, roles=roles, user_text=text))
    if skipped:
        print(f"  [worklist] skipped {skipped} records with unbuildable images", flush=True)
    return work


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(SUITE.parent),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def row_to_record(item: WorkItem, row) -> dict:
    """Merge a backend ResultRow with record provenance into one output dict."""
    rec = item.record
    gold = row.gold if row.gold is not None else suite_parsing.parse_letter(rec.get("answer"))
    out = {
        "index": row.index,
        "question_id": rec.get("question_id"),
        "sample_id": rec.get("sample_id"),
        "topic": rec.get("topic"),
        "difficulty": rec.get("difficulty"),
        "city": rec.get("city"),
        "country": rec.get("country"),
        "city_type": rec.get("city_type"),
        "benchmark_city_type": rec.get("benchmark_city_type"),
        "land_use": rec.get("land_use"),
        "generation_method": rec.get("generation_method"),
        "image_mode": rec.get("image_mode"),
        "question": rec.get("question"),
        "options": rec.get("options"),
        "ablation_mode": row.mode,
        "gold": gold,
        "prediction": row.prediction,
        "is_correct": (gold is not None and row.prediction is not None and gold == row.prediction),
        "prob_dict": row.prob_dict,
        "hedged": row.hedged,
        "refused": row.refused,
        "n_images": row.n_images,
        "image_roles": row.roles,
        "raw_response": row.raw_response,
    }
    if row.attention is not None:
        out["attention"] = row.attention
    return out


def write_model_outputs(out_dir: Path, model_cfg, backend_name, rows_by_mode,
                        meta_extra: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": model_cfg["key"], "display": model_cfg.get("display"),
               "backend": backend_name, "modes": {}}
    all_rows = []
    has_attention = False
    for mode, recs in rows_by_mode.items():
        # predictions
        with open(out_dir / f"{mode}_predictions.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # attention split out (it can be large)
        attn_rows = [r for r in recs if "attention" in r]
        if attn_rows:
            has_attention = True
            with open(out_dir / f"{mode}_attention.jsonl", "w") as f:
                for r in attn_rows:
                    f.write(json.dumps({
                        "question_id": r["question_id"], "ablation_mode": mode,
                        "prediction": r["prediction"], "gold": r["gold"],
                        "is_correct": r["is_correct"], "image_roles": r["image_roles"],
                        "attention": r["attention"],
                    }, ensure_ascii=False) + "\n")
        report = suite_metrics.build_full_report(recs)
        with open(out_dir / f"{mode}_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        summary["modes"][mode] = {"n": len(recs), "overall": report["overall"],
                                  "by_topic": {t: a["accuracy"] for t, a in report["by_topic"].items()}}
        all_rows.extend(recs)
        print(f"    [{mode}] n={len(recs)} acc={report['overall']['accuracy']:.4f}", flush=True)

    with open(out_dir / "all_predictions.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    meta = {**model_cfg, "backend": backend_name, "git_sha": _git_sha(),
            "has_attention": has_attention, **meta_extra}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model_cfg, defaults, records, image_root, out_root, out_key, ds_meta):
    """Run one model against one already-loaded dataset.

    `out_key` is the output subdir name (e.g. "<key>__<dataset_tag>"); a model
    evaluated on several datasets produces one subdir per (model, dataset) so
    results never collide. `ds_meta` is recorded in meta.json for provenance.
    """
    key = model_cfg["key"]
    out_dir = out_root / out_key
    if (out_dir / "summary.json").exists() and not os.environ.get("FORCE"):
        print(f"\n=== {out_key} === SKIP (results/{out_key}/summary.json exists; FORCE=1 to rerun)", flush=True)
        return
    backend_name = resolve_backend(model_cfg)
    requested_modes = model_cfg.get("modes", defaults.get("modes"))
    image_max_edge = model_cfg.get("image_max_edge", defaults.get("image_max_edge", 768))

    # Satellite-only single-image models (GeoChat / SkySenseGPT) are confined to
    # the 9 Family-1 overhead topics and the sat_only mode (the single marked-sat
    # tile): they have one <image> slot, never trained on street-view, and the
    # other topics/modes would feed them multiple or street-level images. This is
    # enforced here, not left to config, so these models can never be accidentally
    # fed an out-of-distribution item.
    only_topics = None
    if model_cfg.get("family") in LLAVA_NATIVE_FAMILIES:
        only_topics = suite_modes.URBAN_ATTRIBUTE_TOPICS
        requested_modes = ["sat_only"]

    print(f"\n=== {out_key} === backend={backend_name} "
          f"attention={bool(model_cfg.get('attention'))} "
          f"lora={'yes' if model_cfg.get('lora_path') else 'no'} "
          f"dataset={ds_meta.get('tag')} ({len(records)} recs)", flush=True)

    work = build_worklist(records, image_root, requested_modes, image_max_edge,
                          only_topics=only_topics)
    print(f"  worklist: {len(work)} (record,mode) items; "
          f"per-mode {dict(Counter(w.mode for w in work))}", flush=True)

    backend = make_backend(backend_name, model_cfg, defaults)
    t0 = time.time()
    rows = backend.run(work)
    backend.close()
    elapsed = time.time() - t0

    rows_by_mode = defaultdict(list)
    for item, row in zip(work, rows):
        rows_by_mode[row.mode].append(row_to_record(item, row))

    write_model_outputs(out_dir, model_cfg, backend_name, rows_by_mode,
                        meta_extra={"n_items": len(work), "elapsed_s": round(elapsed, 1),
                                    "image_max_edge": image_max_edge,
                                    "out_key": out_key, "dataset": ds_meta})
    print(f"  done in {elapsed/60:.1f} min -> {out_dir}", flush=True)


def make_backend(name, model_cfg, defaults):
    if name == "transformers":
        if model_cfg.get("attention"):
            from backends.attention_backend import AttentionBackend
            return AttentionBackend(model_cfg, defaults)
        from backends.hf_backend import HFBackend
        return HFBackend(model_cfg, defaults)
    if name == "vllm":
        from backends.vllm_backend import VLLMBackend
        return VLLMBackend(model_cfg, defaults)
    if name == "llava_native":
        from backends.llava_native_backend import LlavaNativeBackend
        return LlavaNativeBackend(model_cfg, defaults)
    raise ValueError(f"unknown backend {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(SUITE / "config.yaml"))
    ap.add_argument("--models", nargs="+", default=None, help="subset of model keys")
    ap.add_argument("--limit-per-topic", type=int, default=None, help="override smoke cap")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    defaults = cfg.get("defaults", {})
    global_ds = cfg.get("dataset")
    repo = SUITE.parent

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (repo / p)

    limit_override = args.limit_per_topic
    out_root = Path(args.outdir) if args.outdir else (SUITE / "results")

    # A dataset block may be specified globally (cfg["dataset"]) and/or per model
    # (model_cfg["dataset"], which may be a single block or a LIST of blocks).
    # Each (model, dataset) pair is one run, written to results/<key>__<tag>/.
    # Records for a given (path, limit) are loaded once and cached across models.
    _rec_cache: dict = {}

    def _dataset_blocks_for(model_cfg):
        own = model_cfg.get("dataset", global_ds)
        if own is None:
            raise ValueError(
                f"[{model_cfg['key']}] no dataset: define one globally or per-model.")
        return own if isinstance(own, list) else [own]

    def _load_cached(ds_block):
        data_path = _resolve(ds_block["path"])
        image_root = _resolve(ds_block["image_root"])
        limit = (limit_override if limit_override is not None
                 else ds_block.get("limit_per_topic", 0))
        cache_key = (str(data_path), limit)
        if cache_key not in _rec_cache:
            recs = load_records(data_path, limit)
            _rec_cache[cache_key] = recs
            print(f"loaded {len(recs)} records from {data_path}"
                  + (f" (limit_per_topic={limit})" if limit else ""), flush=True)
        tag = ds_block.get("tag") or data_path.stem
        ds_meta = {"tag": tag, "path": str(data_path),
                   "image_root": str(image_root), "limit_per_topic": limit}
        return _rec_cache[cache_key], image_root, tag, ds_meta

    model_list = cfg["models"]
    if args.models:
        wanted = set(args.models)
        model_list = [m for m in model_list if m["key"] in wanted]
        if not model_list:
            print(f"no models matched {args.models}; available: "
                  f"{[m['key'] for m in cfg['models']]}", flush=True)
            return

    for model_cfg in model_list:
        for ds_block in _dataset_blocks_for(model_cfg):
            try:
                records, image_root, tag, ds_meta = _load_cached(ds_block)
                # Single-dataset models keep their bare key as the output dir; a
                # model with multiple datasets gets <key>__<tag> to avoid collision.
                blocks = _dataset_blocks_for(model_cfg)
                out_key = model_cfg["key"] if len(blocks) == 1 else f"{model_cfg['key']}__{tag}"
                run_model(model_cfg, defaults, records, image_root, out_root, out_key, ds_meta)
            except Exception as e:
                print(f"!! {model_cfg['key']} [{ds_block.get('tag','?')}] FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
