"""test_image_parity.py — prove the vendored image builder == training/data.py.

The suite deliberately re-implements image construction (images.build_images)
instead of importing the training package, to stay decoupled. The DANGER of that
choice is silent pixel drift: if our images ever differ from what the model was
trained/evaluated on, every accuracy and attention number is quietly wrong.

This test removes that danger. For a sample of records covering EVERY image_mode
it builds the image list both ways — via benchmark_suite.images.build_images and
via training.data.convert_record — and asserts the resulting PIL images are
byte-for-byte identical (same count, same role order, same pixels).

Run:
    conda activate unsloth
    cd benchmark_suite && python -m tests.test_image_parity
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parent.parent.parent
SUITE = REPO / "benchmark_suite"
sys.path.insert(0, str(SUITE))

# training/data.py needs config importable and TEXT_ONLY off; it reads the same
# dataset dir we do. We import it ONLY here in the test (never in the suite).
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING))

BENCH = REPO / "dataset_content" / "EODATA_compressed_final" / "benchmark"
DATA = BENCH / "benchmark_with_answers.jsonl"
MAX_EDGE = 768
PER_MODE = 2  # records sampled per image_mode


def _load_sample():
    import json
    by_mode = defaultdict(list)
    with open(DATA) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("image_mode", "satellite_only")
            if len(by_mode[m]) < PER_MODE:
                by_mode[m].append(r)
    return by_mode


def _imgs_equal(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    a = a.convert("RGB")
    b = b.convert("RGB")
    return ImageChops.difference(a, b).getbbox() is None


def main():
    import images as suite_images

    # Import training.data with TEXT_ONLY off so it builds images.
    os.environ.setdefault("BASE_MODEL", str(REPO / "models" / "Qwen3.5-4B"))
    os.environ["TEXT_ONLY"] = "0"
    import data as train_data

    by_mode = _load_sample()
    assert by_mode, "no records loaded"

    total = 0
    failures = []
    for mode, recs in sorted(by_mode.items()):
        for rec in recs:
            total += 1
            qid = rec.get("question_id", "?")

            ours = suite_images.build_images(rec, str(BENCH), MAX_EDGE)
            our_imgs = [d["image"] for d in ours]

            converted = train_data.convert_record(rec, str(BENCH), MAX_EDGE)
            user_msg = converted["messages"][1]
            ref_imgs = [p["image"] for p in user_msg["content"] if p["type"] == "image"]

            if len(our_imgs) != len(ref_imgs):
                failures.append(
                    f"[{mode}] {qid}: image COUNT {len(our_imgs)} != {len(ref_imgs)}"
                )
                continue
            for i, (a, b) in enumerate(zip(our_imgs, ref_imgs)):
                if not _imgs_equal(a, b):
                    failures.append(
                        f"[{mode}] {qid}: image #{i} (role={ours[i]['role']}) "
                        f"PIXELS differ (sizes {a.size} vs {b.size})"
                    )

    print(f"checked {total} records across {len(by_mode)} image_modes: "
          f"{sorted(by_mode)}")
    if failures:
        print(f"\nPARITY FAILURES ({len(failures)}):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("PARITY OK — vendored images are byte-identical to training/data.py")


if __name__ == "__main__":
    main()
