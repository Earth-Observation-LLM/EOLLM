#!/usr/bin/env python3
"""
Package dataset splits into self-contained directories.

Each split gets its own folder with images organized by type and a single
JSONL file with updated (local) image paths.

Output structure:
    {outdir}/
        train/
            train.jsonl
            images/
                sat/          ← satellite tiles
                sv/           ← street view (4 per location)
                sat_marked/   ← satellite with center dot
                sat_arrow/    ← satellite with bearing arrows
                composite/    ← 2x2 grids, mismatch composites
        validation/
            validation.jsonl
            images/
                ...  (same subdirs)
        benchmark/
            benchmark_public.jsonl          ← answers nulled
            benchmark_with_answers.jsonl    ← private, has answers
            images/
                ...  (same subdirs)

Usage:
    python splitting/package_splits.py \\
        --splits-dir splits/ \\
        --image-dirs /home/alperen/Documents/EODATA/output_1445/output \\
                     /home/alperen/Documents/EODATA/output_1459_28_03/output \\
                     /home/alperen/Documents/EODATA/output_673img/output \\
                     /home/alperen/Documents/EODATA/output_707/output \\
        --outdir packaged/
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_image_paths(record: dict) -> set[str]:
    """Extract all image paths referenced by a flat question record."""
    paths = set()

    # From the images dict (location-level: sat, sv, composites, marked)
    images = record.get("images", {})
    for key, val in images.items():
        if isinstance(val, str) and val.startswith("images/"):
            paths.add(val)

    # Question-level path fields
    str_fields = [
        "sat_marked_path",
        "query_stv_path",
        "mismatch_negative_stv_composite",
        "composite_stv_path",
        "composite_stv_labeled_path",
        "stv_shown_composite",
    ]
    for field in str_fields:
        val = record.get(field)
        if val and isinstance(val, str) and val.startswith("images/"):
            paths.add(val)

    # List fields
    list_fields = [
        "mismatch_negative_stv_paths",
        "stv_shown_paths",
    ]
    for field in list_fields:
        val = record.get(field)
        if val and isinstance(val, list):
            for p in val:
                if isinstance(p, str) and p.startswith("images/"):
                    paths.add(p)

    # Dict fields: option_sat_paths, option_stv_paths, option_composite_paths
    dict_fields = [
        "option_sat_paths",
        "option_stv_paths",
        "option_composite_paths",
    ]
    for field in dict_fields:
        val = record.get(field)
        if val and isinstance(val, dict):
            for opt_key, opt_val in val.items():
                if isinstance(opt_val, str) and opt_val.startswith("images/"):
                    paths.add(opt_val)
                elif isinstance(opt_val, list):
                    for p in opt_val:
                        if isinstance(p, str) and p.startswith("images/"):
                            paths.add(p)

    return paths


def find_source(image_path: str, image_dirs: list[str]) -> str | None:
    """Find the actual file across multiple source directories."""
    for base in image_dirs:
        full = os.path.join(base, image_path)
        if os.path.exists(full):
            return full
    return None


def copy_images(
    image_paths: set[str],
    image_dirs: list[str],
    dest_base: str,
) -> tuple[int, int]:
    """Copy images from source dirs to destination, preserving relative paths.

    Returns:
        (copied_count, missing_count)
    """
    copied = 0
    missing = 0

    for rel_path in sorted(image_paths):
        dest = os.path.join(dest_base, rel_path)

        # Skip if already copied
        if os.path.exists(dest):
            copied += 1
            continue

        source = find_source(rel_path, image_dirs)
        if source is None:
            missing += 1
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(source, dest)
        copied += 1

    return copied, missing


def package_split(
    split_name: str,
    records: list[dict],
    image_dirs: list[str],
    outdir: str,
    include_private: bool = False,
):
    """Package a single split into a self-contained directory."""
    split_dir = os.path.join(outdir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Collect all image paths
    all_paths = set()
    for rec in records:
        all_paths |= collect_image_paths(rec)

    print(f"  [{split_name}] {len(records)} questions, {len(all_paths)} unique images")

    # Copy images
    copied, missing = copy_images(all_paths, image_dirs, split_dir)
    print(f"  [{split_name}] Copied: {copied}, Missing: {missing}")
    if missing > 0:
        print(f"  [{split_name}] WARNING: {missing} images not found in any source directory")

    # Write JSONL (paths remain as-is since images/ is now local)
    if split_name == "benchmark":
        # Public version: null out answers
        public_records = []
        for rec in records:
            pub = dict(rec)
            pub["answer"] = None
            public_records.append(pub)
        save_jsonl(public_records, os.path.join(split_dir, "benchmark_public.jsonl"))
        print(f"  [{split_name}] Wrote benchmark_public.jsonl ({len(public_records)} records)")

        if include_private:
            save_jsonl(records, os.path.join(split_dir, "benchmark_with_answers.jsonl"))
            print(f"  [{split_name}] Wrote benchmark_with_answers.jsonl ({len(records)} records)")
    else:
        save_jsonl(records, os.path.join(split_dir, f"{split_name}.jsonl"))
        print(f"  [{split_name}] Wrote {split_name}.jsonl ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(
        description="Package dataset splits into self-contained directories with images"
    )
    parser.add_argument(
        "--splits-dir",
        default="splits",
        help="Directory containing split JSONL files (default: splits/)",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        required=True,
        help="Source directories containing output/images/ (searched in order)",
    )
    parser.add_argument(
        "--outdir",
        default="packaged",
        help="Output directory for packaged splits (default: packaged/)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load splits
    splits = {
        "train": os.path.join(args.splits_dir, "train.jsonl"),
        "validation": os.path.join(args.splits_dir, "validation.jsonl"),
        "benchmark": os.path.join(args.splits_dir, "benchmark_with_answers.jsonl"),
    }

    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run split_dataset.py first.")
            sys.exit(1)

    print("Packaging splits...\n")

    for name, path in splits.items():
        records = load_jsonl(path)
        package_split(
            split_name=name,
            records=records,
            image_dirs=args.image_dirs,
            outdir=args.outdir,
            include_private=(name == "benchmark"),
        )
        print()

    print(f"Done! Packaged splits in {args.outdir}/")
    print(f"  {args.outdir}/train/          ← training set (with answers)")
    print(f"  {args.outdir}/validation/     ← validation set (with answers)")
    print(f"  {args.outdir}/benchmark/      ← benchmark (public has nulled answers)")


if __name__ == "__main__":
    main()
