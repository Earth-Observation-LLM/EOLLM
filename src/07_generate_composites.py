"""
Step 7: Generate composites and mismatch metadata for geolocation tasks.

Creates:
  - Red-dot marked satellite images (for all geolocation question types)
  - Per-option STV composites (2x2 of 4 angles per location)
  - Overall STV option grids (2x2 of per-option composites) for mismatch_mcq
  - Negative pair metadata for mismatch_binary

Run after satellite & street view fetch (steps 2-3), before question
generation (step 5).
"""

import os
import math
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# STV angle labels in composite layout order:
# Top-Left=fwd, Top-Right=right, Bottom-Left=bwd, Bottom-Right=left
STV_ANGLES = ["along_fwd", "cross_right", "along_bwd", "cross_left"]
POS_LABELS = ["A", "B", "C", "D"]


def _find_in_dirs(rel_path, source_dirs):
    """Search for a relative image path across multiple source directories."""
    for d in source_dirs:
        full = os.path.join(d, rel_path)
        if os.path.exists(full):
            return full
    return None


def _get_sat_path(sample, source_dirs=None):
    """Get satellite image path for a sample."""
    rel = os.path.join("images", "sat", f"{sample['sample_id']}.png")
    if source_dirs:
        return _find_in_dirs(rel, source_dirs) or ""
    return os.path.join(ROOT, "output", rel)


def _get_stv_paths(sample, source_dirs=None):
    """Get all 4 STV angle paths for a sample."""
    sid = sample["sample_id"]
    paths = []
    for a in STV_ANGLES:
        rel = os.path.join("images", "sv", f"{sid}_{a}.jpg")
        if source_dirs:
            paths.append(_find_in_dirs(rel, source_dirs) or "")
        else:
            paths.append(os.path.join(ROOT, "output", rel))
    return paths


def _stv_exists(sample, source_dirs=None):
    """Check if at least the forward STV exists."""
    sid = sample["sample_id"]
    rel = os.path.join("images", "sv", f"{sid}_along_fwd.jpg")
    if source_dirs:
        return _find_in_dirs(rel, source_dirs) is not None
    return os.path.exists(os.path.join(ROOT, "output", rel))


def _all_stv_exist(sample, source_dirs=None):
    """Check if all 4 STV angles exist."""
    return all(os.path.exists(p) for p in _get_stv_paths(sample, source_dirs))


def _relative_stv_paths(sample):
    """Return relative STV paths (for JSONL storage)."""
    sid = sample["sample_id"]
    return [f"images/sv/{sid}_{a}.jpg" for a in STV_ANGLES]


def tile_2x2(image_paths, output_path, tile_size=256):
    """Create a 2x2 grid from 4 image paths."""
    from PIL import Image
    canvas = Image.new('RGB', (tile_size * 2, tile_size * 2))
    for i, path in enumerate(image_paths):
        img = Image.open(path).resize((tile_size, tile_size))
        x = (i % 2) * tile_size
        y = (i // 2) * tile_size
        canvas.paste(img, (x, y))
    canvas.save(output_path)
    return output_path


def tile_2x2_labeled(image_paths, output_path, labels, tile_size=256):
    """Create a 2x2 grid with labels burned into each quadrant."""
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (tile_size * 2, tile_size * 2))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, path in enumerate(image_paths):
        img = Image.open(path).resize((tile_size, tile_size))
        x = (i % 2) * tile_size
        y = (i // 2) * tile_size
        canvas.paste(img, (x, y))
        # Draw label with background
        label = labels[i] if i < len(labels) else ""
        if label:
            tx, ty = x + 8, y + 8
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
                           fill=(0, 0, 0, 180))
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    canvas.save(output_path)
    return output_path


def create_red_dot_image(sat_path, output_path, dot_radius=8):
    """Overlay a red dot at the center of a satellite image."""
    from PIL import Image, ImageDraw
    img = Image.open(sat_path).copy()
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    # White outline ring
    draw.ellipse([cx - dot_radius - 2, cy - dot_radius - 2,
                  cx + dot_radius + 2, cy + dot_radius + 2],
                 fill=(255, 255, 255))
    # Red filled circle
    draw.ellipse([cx - dot_radius, cy - dot_radius,
                  cx + dot_radius, cy + dot_radius],
                 fill=(255, 0, 0))
    img.save(output_path)
    return output_path


def create_arrow_image(sat_path, output_path, bearing_deg,
                       dot_radius=8, arrow_length=80):
    """Overlay a red dot and directional arrow on a satellite image.

    bearing_deg: compass bearing (0=N, 90=E, 180=S, 270=W).
    Arrow starts at image center and points in the bearing direction.
    """
    from PIL import Image, ImageDraw
    img = Image.open(sat_path).copy()
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2

    # Red dot (same as create_red_dot_image)
    draw.ellipse([cx - dot_radius - 2, cy - dot_radius - 2,
                  cx + dot_radius + 2, cy + dot_radius + 2],
                 fill=(255, 255, 255))
    draw.ellipse([cx - dot_radius, cy - dot_radius,
                  cx + dot_radius, cy + dot_radius],
                 fill=(255, 0, 0))

    # Arrow endpoint (image coords: y increases downward)
    rad = math.radians(bearing_deg)
    ex = cx + math.sin(rad) * arrow_length
    ey = cy - math.cos(rad) * arrow_length

    # Arrow shaft
    draw.line([(cx, cy), (ex, ey)], fill=(255, 0, 0), width=4)

    # Arrowhead triangle
    head_len = 16
    head_half = 8
    # Unit vector along arrow
    dx = ex - cx
    dy = ey - cy
    length = math.hypot(dx, dy)
    if length > 0:
        ux, uy = dx / length, dy / length
        # Perpendicular
        px, py = -uy, ux
        # Three points of arrowhead
        tip = (ex, ey)
        left = (ex - ux * head_len + px * head_half,
                ey - uy * head_len + py * head_half)
        right = (ex - ux * head_len - px * head_half,
                 ey - uy * head_len - py * head_half)
        draw.polygon([tip, left, right], fill=(255, 0, 0))

    img.save(output_path)
    return output_path


def _build_stv_composite(sample, composite_dir, suffix="", source_dirs=None):
    """Build a 2x2 composite of all 4 STV angles for one sample.

    Returns (clean_path, labeled_path) as relative paths, or (None, None).
    """
    sid = sample["sample_id"]
    stv_paths = _get_stv_paths(sample, source_dirs)
    if not all(p and os.path.exists(p) for p in stv_paths):
        return None, None

    clean_name = f"{sid}_4angles{suffix}.png"
    labeled_name = f"{sid}_4angles{suffix}_labeled.png"
    clean_path = os.path.join(composite_dir, clean_name)
    labeled_path = os.path.join(composite_dir, labeled_name)

    tile_2x2(stv_paths, clean_path)
    angle_labels = ["Fwd", "Right", "Bwd", "Left"]
    tile_2x2_labeled(stv_paths, labeled_path, angle_labels)

    return f"images/composite/{clean_name}", f"images/composite/{labeled_name}"


def run(samples, output_base=None, source_dirs=None):
    """Generate red-dot images, STV composites, and mismatch metadata.

    Args:
        samples: List of sample dicts.
        output_base: Base directory for writing generated images (default: ROOT/output).
        source_dirs: List of directories to search for source sat/sv images.
                     Default: [ROOT/output]. Useful when source images are spread
                     across multiple collection runs.
    """
    from config import MISMATCH_MCQ_STRATEGY, MISMATCH_BINARY_STRATEGY

    if output_base is None:
        output_base = os.path.join(ROOT, "output")
    if source_dirs is None:
        source_dirs = [os.path.join(ROOT, "output")]

    print(f"[Composites] Generating composites (output: {output_base})...")

    composite_dir = os.path.join(output_base, "images", "composite")
    marked_dir = os.path.join(output_base, "images", "sat_marked")
    arrow_dir = os.path.join(output_base, "images", "sat_arrow")
    os.makedirs(composite_dir, exist_ok=True)
    os.makedirs(marked_dir, exist_ok=True)
    os.makedirs(arrow_dir, exist_ok=True)

    # Pre-filter samples with valid images
    samples_with_sat = [s for s in samples
                        if os.path.exists(_get_sat_path(s, source_dirs))]
    samples_with_stv = [s for s in samples
                        if _all_stv_exist(s, source_dirs)]

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        sat_path = _get_sat_path(sample, source_dirs)

        if not sat_path or not os.path.exists(sat_path):
            continue

        # ── Red dot image ──────────────────────────────────────────────
        marked_path = os.path.join(marked_dir, f"{sid}.png")
        try:
            create_red_dot_image(sat_path, marked_path)
            sample["sat_marked_path"] = f"images/sat_marked/{sid}.png"
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {sid}: red dot FAILED: {e}")

        # ── Arrow images for camera_direction ────────────────────────
        bearing = sample.get("road_bearing")
        if bearing is not None:
            try:
                bearing = float(bearing)
                angle_map = {
                    "fwd": bearing % 360,
                    "right": (bearing - 90) % 360,
                    "bwd": (bearing + 180) % 360,
                    "left": (bearing + 90) % 360,
                }
                arrow_paths = {}
                for label, hdg in angle_map.items():
                    arrow_out = os.path.join(arrow_dir, f"{sid}_arrow_{label}.png")
                    create_arrow_image(sat_path, arrow_out, hdg)
                    arrow_paths[f"along_{label}" if label in ("fwd", "bwd")
                                else f"cross_{label}"] = \
                        f"images/sat_arrow/{sid}_arrow_{label}.png"
                sample["camera_arrow_paths"] = arrow_paths
            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] {sid}: arrows FAILED: {e}")

        if not _all_stv_exist(sample, source_dirs):
            continue

        # ── Per-sample STV composite (own 4 angles) ───────────────────
        own_clean, own_labeled = _build_stv_composite(
            sample, composite_dir, source_dirs=source_dirs)
        if own_clean:
            sample["stv_composite_path"] = own_clean
            sample["stv_composite_labeled_path"] = own_labeled

        # ── Distractor pools (only samples in THIS run) ──────────────
        others = [s for s in samples_with_stv if s["sample_id"] != sid]
        same_city = [s for s in others if s["city"] == sample["city"]]
        cross_city = [s for s in others if s["city"] != sample["city"]]

        rng = random.Random(sid)

        # Determine which strategies to run
        strategies = []
        if MISMATCH_MCQ_STRATEGY in ("same_city", "both"):
            strategies.append("same_city")
        if MISMATCH_MCQ_STRATEGY in ("cross_city", "both"):
            strategies.append("cross_city")

        # ── Mismatch MCQ variants ─────────────────────────────────────
        mcq_variants = []
        for strategy in strategies:
            pool = same_city if strategy == "same_city" else cross_city
            if len(pool) < 3:
                continue

            chosen = rng.sample(pool, 3)
            all_option_samples = [sample] + chosen  # index 0 = correct

            # Shuffle and track correct position
            indices = list(range(4))
            rng.shuffle(indices)
            correct_pos = indices.index(0)

            # Build per-option composites for each location
            option_stv_paths = {}
            option_composite_paths = {}
            per_option_clean_paths = []

            for pos_idx, orig_idx in enumerate(indices):
                opt_sample = all_option_samples[orig_idx]
                opt_label = POS_LABELS[pos_idx]

                # Individual STV paths
                option_stv_paths[opt_label] = _relative_stv_paths(opt_sample)

                # Per-option 4-angle composite
                opt_suffix = f"_{sid}_opt{opt_label}_{strategy}"
                opt_clean, opt_labeled = _build_stv_composite(
                    opt_sample, composite_dir, suffix=opt_suffix,
                    source_dirs=source_dirs)
                option_composite_paths[opt_label] = opt_clean

                if opt_clean:
                    per_option_clean_paths.append(
                        os.path.join(output_base, opt_clean))
                else:
                    per_option_clean_paths.append(None)

            # Build overall 2x2 mega-composite (of per-option composites)
            if all(p and os.path.exists(p) for p in per_option_clean_paths):
                mega_name = f"{sid}_4stv_{strategy}.png"
                mega_labeled_name = f"{sid}_4stv_{strategy}_labeled.png"
                mega_path = os.path.join(composite_dir, mega_name)
                mega_labeled_path = os.path.join(composite_dir, mega_labeled_name)

                try:
                    tile_2x2(per_option_clean_paths, mega_path)
                    tile_2x2_labeled(per_option_clean_paths, mega_labeled_path,
                                     POS_LABELS)

                    variant = {
                        "strategy": strategy,
                        "composite_path": f"images/composite/{mega_name}",
                        "composite_labeled_path": f"images/composite/{mega_labeled_name}",
                        "correct_pos": POS_LABELS[correct_pos],
                        "distractor_sids": [c["sample_id"] for c in chosen],
                        "difficulty": "hard" if strategy == "same_city" else "easy",
                        "option_stv_paths": option_stv_paths,
                        "option_composite_paths": option_composite_paths,
                    }
                    mcq_variants.append(variant)
                    print(f"  [{i+1}/{len(samples)}] {sid}: MCQ {strategy} "
                          f"(correct={POS_LABELS[correct_pos]})")
                except Exception as e:
                    print(f"  [{i+1}/{len(samples)}] {sid}: MCQ {strategy} FAILED: {e}")

        if mcq_variants:
            sample["mismatch_mcq_variants"] = mcq_variants

        # ── Mismatch binary variants ──────────────────────────────────
        bin_strategies = []
        if MISMATCH_BINARY_STRATEGY in ("same_city", "both"):
            bin_strategies.append("same_city")
        if MISMATCH_BINARY_STRATEGY in ("cross_city", "both"):
            bin_strategies.append("cross_city")

        binary_variants = []
        for strategy in bin_strategies:
            pool = same_city if strategy == "same_city" else cross_city
            if not pool:
                continue

            neg_sample = rng.choice(pool)
            neg_sid = neg_sample["sample_id"]

            # Build composite for the negative sample if not already done
            neg_suffix = f"_{sid}_neg_{strategy}"
            neg_clean, neg_labeled = _build_stv_composite(
                neg_sample, composite_dir, suffix=neg_suffix,
                source_dirs=source_dirs)

            variant = {
                "strategy": strategy,
                "negative_sid": neg_sid,
                "negative_stv_paths": _relative_stv_paths(neg_sample),
                "negative_stv_composite": neg_clean,
                "negative_stv_composite_labeled": neg_labeled,
                "difficulty": "hard" if strategy == "same_city" else "easy",
            }
            binary_variants.append(variant)
            print(f"  [{i+1}/{len(samples)}] {sid}: Binary {strategy} "
                  f"(neg={neg_sid})")

        if binary_variants:
            sample["mismatch_binary_variants"] = binary_variants

    # Summary
    mcq_count = sum(1 for s in samples if s.get("mismatch_mcq_variants"))
    bin_count = sum(1 for s in samples if s.get("mismatch_binary_variants"))
    dot_count = sum(1 for s in samples if s.get("sat_marked_path"))
    arrow_count = sum(1 for s in samples if s.get("camera_arrow_paths"))
    print(f"  Red dots: {dot_count}, Arrows: {arrow_count}, "
          f"MCQ variants: {mcq_count}, Binary variants: {bin_count}")
    return samples


if __name__ == "__main__":
    import pandas as pd
    csv_path = os.path.join(ROOT, "output", "metadata_raw.csv")
    samples = pd.read_csv(csv_path).to_dict('records')
    run(samples)
