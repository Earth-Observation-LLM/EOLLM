#!/usr/bin/env python3
"""
Merge image directories from multiple split folders into a single shared directory.

Moves (not copies) images from source dirs into one combined images/ folder,
so you can generate different JSONL splits without duplicating images.

Usage:
    python splitting/merge_images.py \
        --source-dirs /mnt/disk2/train /mnt/disk2/validation \
        --dest /mnt/disk2/all_images

This will:
  1. Move images/ from each source dir into --dest/images/
  2. Skip files that already exist in dest (by filename)
  3. Print a summary of moved/skipped/missing files

After merging, all JSONL files can reference images as "images/sat/..." relative
to the --dest directory regardless of which split they belong to.
"""

import argparse
import os
import shutil


def merge_image_dirs(source_dirs: list[str], dest: str, dry_run: bool = False):
    """Merge images/ subdirectories from multiple sources into one destination.

    Args:
        source_dirs: List of directories, each expected to contain an images/ subfolder.
        dest: Destination directory. images/ will be created here.
        dry_run: If True, only print what would happen without moving files.
    """
    dest_images = os.path.join(dest, "images")
    os.makedirs(dest_images, exist_ok=True)

    total_moved = 0
    total_skipped = 0
    total_missing = 0

    for src_dir in source_dirs:
        src_images = os.path.join(src_dir, "images")
        if not os.path.isdir(src_images):
            print(f"  WARNING: {src_images} does not exist, skipping")
            total_missing += 1
            continue

        print(f"\n  Processing: {src_images}")
        moved = 0
        skipped = 0

        for root, dirs, files in os.walk(src_images):
            # Compute the relative path from src_images
            rel_dir = os.path.relpath(root, src_images)
            dest_subdir = os.path.join(dest_images, rel_dir)

            if not dry_run:
                os.makedirs(dest_subdir, exist_ok=True)

            for fname in files:
                src_file = os.path.join(root, fname)
                dest_file = os.path.join(dest_subdir, fname)

                if os.path.exists(dest_file):
                    skipped += 1
                    continue

                if dry_run:
                    print(f"    Would move: {os.path.join(rel_dir, fname)}")
                else:
                    shutil.move(src_file, dest_file)
                moved += 1

        print(f"    Moved: {moved}, Skipped (already exists): {skipped}")
        total_moved += moved
        total_skipped += skipped

    print(f"\n  Total moved: {total_moved}")
    print(f"  Total skipped: {total_skipped}")
    if total_missing:
        print(f"  Source dirs without images/: {total_missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge image directories from split folders into one shared directory"
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="Split directories containing images/ subfolders (e.g., packaged/train packaged/validation)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for merged images/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be moved without actually moving files",
    )
    args = parser.parse_args()

    print("Merging image directories...")
    merge_image_dirs(args.source_dirs, args.dest, args.dry_run)

    print(f"\nDone! All images now in {args.dest}/images/")
    print("JSONL files can reference images as 'images/...' relative to this directory.")


if __name__ == "__main__":
    main()
