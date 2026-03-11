"""
Step 6: Validate all samples — per-sample and dataset-level checks.
"""

import os
import json
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate_sample(sample):
    """Run validation checks on a single sample."""
    issues = []

    # Answer in options
    answer = sample.get("answer")
    options = sample.get("options", {})
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            try:
                import ast
                options = ast.literal_eval(options)
            except Exception:
                options = {}

    if not sample.get("question"):
        issues.append("CRITICAL: No question generated")
    if answer and answer not in options:
        issues.append("CRITICAL: Answer key not in options")
    if options:
        vals = list(options.values())
        if len(set(vals)) != len(vals):
            issues.append("CRITICAL: Duplicate options")
        if len(vals) != 4:
            issues.append(f"WARNING: Expected 4 options, got {len(vals)}")

    # Satellite image — missing or tiny images are CRITICAL since the model
    # cannot learn from blank/corrupt imagery
    sat_path = os.path.join(ROOT, "output", "images", "sat",
                            f"{sample['sample_id']}.png")
    if not os.path.exists(sat_path):
        issues.append("CRITICAL: Missing satellite image")
    elif os.path.getsize(sat_path) < 5000:
        issues.append("CRITICAL: Satellite image too small (likely blank/cloud)")

    # Street view images
    sv_labels = ["along_fwd", "along_bwd", "cross_left", "cross_right"]
    sv_count = 0
    for label in sv_labels:
        sv_path = os.path.join(ROOT, "output", "images", "sv",
                               f"{sample['sample_id']}_{label}.jpg")
        if os.path.exists(sv_path) and os.path.getsize(sv_path) > 5000:
            sv_count += 1
        else:
            issues.append(f"WARNING: Missing/small SV image: {label}")
    if sv_count == 0:
        issues.append("CRITICAL: No street view images at all")

    # OSM metadata sanity
    levels = sample.get("osm_median_levels")
    if levels is not None:
        try:
            lv = float(levels)
            if lv < 0 or lv > 200:
                issues.append(f"WARNING: Implausible building levels: {lv}")
        except (ValueError, TypeError):
            pass

    is_valid = not any("CRITICAL" in i for i in issues)
    return is_valid, issues


def validate_dataset(samples):
    """Run dataset-level checks."""
    issues = []

    # Answer distribution
    answers = [s.get("answer") for s in samples if s.get("answer")]
    if answers:
        ans_dist = Counter(answers)
        if max(ans_dist.values()) > len(samples) * 0.4:
            issues.append(f"WARNING: Answer distribution skewed: {dict(ans_dist)}")

    # Topic diversity
    topics = [s.get("topic") for s in samples if s.get("topic")]
    if len(set(topics)) < 3:
        issues.append(f"WARNING: Low topic diversity: {Counter(topics)}")

    # City coverage
    cities = [s.get("city") for s in samples if s.get("city")]
    if len(set(cities)) < 2:
        issues.append(f"WARNING: Only 1 city represented")

    # Land use diversity
    lus = [s.get("land_use_category") for s in samples if s.get("land_use_category")]
    if len(set(lus)) < 3:
        issues.append(f"WARNING: Low land use diversity: {set(lus)}")

    return issues


def run(samples=None):
    """Main entry point.

    Validates all samples and REMOVES those with CRITICAL issues from the
    returned list. This ensures the final dataset.jsonl only contains
    usable training samples.
    """
    print("[Step 6/6] Validating dataset...")

    if samples is None:
        import pandas as pd
        csv_path = os.path.join(ROOT, "output", "metadata_raw.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    valid_samples = []
    rejected_samples = []
    all_issues = []

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        is_valid, issues = validate_sample(sample)
        sample["validation_valid"] = is_valid
        sample["validation_issues"] = issues

        status = "PASS" if is_valid else "FAIL"
        print(f"  [{i+1}/{len(samples)}] {sid}: {status}")
        for issue in issues:
            print(f"    - {issue}")

        if is_valid:
            valid_samples.append(sample)
        else:
            rejected_samples.append(sample)
        all_issues.extend(issues)

    ds_issues = validate_dataset(valid_samples)
    if ds_issues:
        print("\n  Dataset-level issues:")
        for issue in ds_issues:
            print(f"    - {issue}")
        all_issues.extend(ds_issues)

    crit = sum(1 for i in all_issues if "CRITICAL" in i)
    warn = sum(1 for i in all_issues if "WARNING" in i)
    info = sum(1 for i in all_issues if "INFO" in i)
    print(f"\n  Summary: {len(valid_samples)}/{len(samples)} valid, "
          f"{len(rejected_samples)} rejected "
          f"({crit} critical, {warn} warnings, {info} info)")
    if rejected_samples:
        print(f"  Rejected samples: "
              f"{[s['sample_id'] for s in rejected_samples]}")
    return valid_samples


if __name__ == "__main__":
    run()
