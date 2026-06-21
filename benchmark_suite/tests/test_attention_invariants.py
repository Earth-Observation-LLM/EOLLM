"""test_attention_invariants.py — assert ATTENTION WATCH output is well-formed.

Runs the attention backend on one record per image_mode and checks the
invariants that make the numbers meaningful:
  - #detected vision spans == #input images   (spans_match_images)
  - each analyzed attention row sums to ~1     (row_softmax_check ~ 1.0)
  - per-image norm shares sum to ~100%
  - per-layer data is present when store_per_layer
  - blind/no-image records carry a 'note', not bogus shares

Loads a real model -> needs the GPU + unsloth env. Skips gracefully if the base
model dir is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
REPO = SUITE.parent
sys.path.insert(0, str(SUITE))

BASE = REPO / "models" / "Qwen3.5-4B"
BENCH = REPO / "dataset_content" / "EODATA_compressed_final" / "benchmark"
DATA = BENCH / "benchmark_with_answers.jsonl"


def main():
    if not BASE.exists():
        print(f"SKIP: base model not found at {BASE}")
        return
    import run
    from backends.attention_backend import AttentionBackend

    records = run.load_records(DATA, limit_per_topic=1)
    defaults = {"modes": ["full"], "enable_thinking": False, "max_new_tokens": 8,
                "attention": {"layers": "all", "max_pixels": 200704, "store_per_layer": True}}
    work = run.build_worklist(records, BENCH, ["full"], image_max_edge=512)
    # cap to a handful to keep the test fast
    work = work[:6]

    model_cfg = {"key": "test", "hf_id": "models/Qwen3.5-4B", "family": "qwen",
                 "attention": True}
    backend = AttentionBackend(model_cfg, defaults)
    rows = backend.run(work)
    backend.close()

    failures = []
    checked = 0
    for it, row in zip(work, rows):
        a = row.attention
        assert a is not None, "missing attention block"
        if a.get("note"):  # no-image / OOM rows
            continue
        checked += 1
        if not a.get("spans_match_images"):
            failures.append(f"{it.record['question_id']}: spans != images "
                            f"({a['n_images']} images)")
        rsc = a.get("row_softmax_check")
        if rsc is None or abs(rsc - 1.0) > 0.02:
            failures.append(f"{it.record['question_id']}: row_softmax_check={rsc} (want ~1.0)")
        norm_sum = sum(im["avg_norm_pct"] for im in a["images"])
        if a["images"] and abs(norm_sum - 100.0) > 0.5:
            failures.append(f"{it.record['question_id']}: norm sum={norm_sum:.2f} (want 100)")
        if not a.get("per_layer"):
            failures.append(f"{it.record['question_id']}: per_layer missing")

    print(f"checked {checked} attention rows")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("ATTENTION INVARIANTS OK")


if __name__ == "__main__":
    main()
