"""
Step 7: Generate composite satellite images for mismatch MCQ tasks.

Creates 2x2 tiled satellite images (1 correct + 3 distractor locations)
and prepares binary mismatch metadata. Run after satellite fetch (step 2),
before question generation (step 5).
"""

import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_sat_path(sample):
    """Get satellite image path for a sample."""
    return os.path.join(ROOT, "output", "images", "sat",
                        f"{sample['sample_id']}.png")


def tile_2x2(image_paths, output_path, tile_size=512):
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


def run(samples):
    """Generate composite images and mismatch metadata for cross-view tasks."""
    print("[Composites] Generating satellite composites for mismatch tasks...")

    composite_dir = os.path.join(ROOT, "output", "images", "composite")
    os.makedirs(composite_dir, exist_ok=True)

    # Filter samples that have satellite images
    samples_with_sat = [s for s in samples
                        if os.path.exists(_get_sat_path(s))]

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        sat_path = _get_sat_path(sample)

        if not os.path.exists(sat_path):
            continue

        # Pick 3 distractor samples (prefer same city for harder task)
        others = [s for s in samples_with_sat
                  if s["sample_id"] != sid]
        same_city = [s for s in others if s["city"] == sample["city"]]
        distractor_pool = same_city if len(same_city) >= 3 else others

        rng = random.Random(sid)
        if len(distractor_pool) < 3:
            continue

        chosen = rng.sample(distractor_pool, 3)

        # Store a random "wrong" satellite for binary mismatch
        sample["mismatch_negative_sid"] = chosen[0]["sample_id"]

        # Create 4-SAT composite (1 correct + 3 distractors)
        sat_paths = [sat_path] + [_get_sat_path(d) for d in chosen]
        # Verify all paths exist
        if not all(os.path.exists(p) for p in sat_paths):
            continue

        # Shuffle and track correct position
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)  # where the original ended up
        shuffled_paths = [sat_paths[idx] for idx in indices]

        composite_path = os.path.join(composite_dir, f"{sid}_4sat.png")
        try:
            tile_2x2(shuffled_paths, composite_path)
            pos_labels = ["A", "B", "C", "D"]
            sample["composite_4sat_path"] = f"images/composite/{sid}_4sat.png"
            sample["composite_4sat_correct_pos"] = pos_labels[correct_pos]
            print(f"  [{i+1}/{len(samples)}] {sid}: composite created "
                  f"(correct={pos_labels[correct_pos]})")
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {sid}: composite FAILED: {e}")

    created = sum(1 for s in samples if s.get("composite_4sat_path"))
    print(f"  Created {created}/{len(samples)} composites")
    return samples


if __name__ == "__main__":
    import pandas as pd
    csv_path = os.path.join(ROOT, "output", "metadata_raw.csv")
    samples = pd.read_csv(csv_path).to_dict('records')
    run(samples)
