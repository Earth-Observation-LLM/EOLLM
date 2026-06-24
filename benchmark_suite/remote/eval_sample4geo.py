#!/usr/bin/env python3
"""Sample4Geo cross-view retrieval baseline on the EOLLM mismatch family.

Sample4Geo (Deuser et al., ICCV'23) is a weight-shared Siamese ConvNeXt trained
with InfoNCE for street<->satellite matching. It produces image *embeddings*; we
match by cosine similarity. There are NO MCQ letters in the model -- we pick the
candidate whose street embedding is closest to the satellite query embedding.

Scope: ONLY the 4 mismatch topics, on the labeled split. Each mismatch question
has a satellite query (images.satellite -- the original tile; the red dot used by
the VQA prompt is irrelevant to an embedding model) and one or more street-view
candidate *sets* (4 perspective JPGs each).

ADAPTATION CHOICES (documented; they affect the number):
  * Street aggregation = STITCH. The 4 perspective JPGs (along_fwd=0deg,
    cross_right=90deg, along_bwd=180deg, cross_left=270deg relative to road
    bearing) are horizontally concatenated in panorama order fwd->right->bwd->left
    to approximate the 360deg panorama Sample4Geo was trained on, then resized to
    the model's ground input (384 x 768, i.e. a 1:2 panorama) and embedded once.
  * Satellite query = images.satellite, resized to 384x384 (model's sat input).
  * Both branches share ONE backbone (Siamese) -> same forward for sat & ground.
  * Preprocessing = the repo's get_transforms_val (Resize + ImageNet normalize),
    bicubic/linear resize, no augmentation.
  * mcq: predicted = candidate option with the highest cosine similarity (argmax).
  * binary: predicted = "match" if cosine(sat, street) > THRESHOLD. THRESHOLD is
    calibrated by sweeping for best balanced accuracy on a stratified held-out
    slice (--calib_frac) of the binary questions; the chosen value is reported and
    the SAME value scores the rest. This is a domain-transfer baseline: Sample4Geo
    was trained on CVUSA/VIGOR street<->aerial, not on EOLLM imagery.

Usage:
  eval_sample4geo.py --ckpt <path> --tag vigor_cross \
      --dataset_dir <DATASET_DIR> --outdir results_sample4geo [--limit N] [--topics ...]
"""
import argparse, json, os, sys, time, math
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

MISMATCH_TOPICS = [
    "mismatch_binary_easy", "mismatch_binary_hard",
    "mismatch_mcq_easy", "mismatch_mcq_hard",
]
# panorama order: along_fwd(0) -> cross_right(90) -> along_bwd(180) -> cross_left(270)
PANO_ORDER = ["along_fwd", "cross_right", "along_bwd", "cross_left"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAT_SIZE = (384, 384)            # (H, W)
GROUND_SIZE = (384, 768)         # (H, W) -- 1:2 panorama, matches repo eval config


def load_model(ckpt, repo_dir, img_size=384):
    sys.path.insert(0, repo_dir)
    from sample4geo.model import TimmModel
    m = TimmModel("convnext_base.fb_in22k_ft_in1k_384", pretrained=False, img_size=img_size)
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    return m.eval().cuda().to(torch.float32)


def _norm_chw(img_rgb_uint8, size_hw):
    """Resize (linear) + ImageNet-normalize -> CHW float tensor."""
    h, w = size_hw
    img = cv2.resize(img_rgb_uint8, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def load_sat_tensor(path):
    img = np.array(Image.open(path).convert("RGB"))
    return _norm_chw(img, SAT_SIZE)


def load_ground_tensor(angle_paths):
    """Stitch 4 angle JPGs (in PANO_ORDER) into one wide pano, then resize to ground size."""
    parts = []
    for p in angle_paths:
        im = np.array(Image.open(p).convert("RGB"))
        parts.append(im)
    # resize each part to a common height first so hconcat is valid
    target_h = min(im.shape[0] for im in parts)
    resized = []
    for im in parts:
        scale = target_h / im.shape[0]
        new_w = max(1, int(round(im.shape[1] * scale)))
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_LINEAR_EXACT))
    pano = np.concatenate(resized, axis=1)   # left->right = fwd,right,bwd,left
    return _norm_chw(pano, GROUND_SIZE)


def angle_paths_for_sid(images_dict, base, prefix_paths=None):
    """Build the 4 ordered angle paths. prefix_paths: explicit list (negative/option),
    else derive from images_dict streetview_* keys."""
    if prefix_paths:
        # order the given list by PANO_ORDER suffix
        by_angle = {}
        for p in prefix_paths:
            for a in PANO_ORDER:
                if p.endswith(f"_{a}.jpg"):
                    by_angle[a] = p
        ordered = [by_angle[a] for a in PANO_ORDER if a in by_angle]
        if len(ordered) != 4:  # fall back to given order
            ordered = list(prefix_paths)
        return [os.path.join(base, p) for p in ordered]
    # derive from images dict (the question's own street views)
    ordered = []
    for a in PANO_ORDER:
        key = f"streetview_{a}"
        if key in images_dict:
            ordered.append(os.path.join(base, images_dict[key]))
    return ordered


@torch.no_grad()
def embed_batch(model, tensors):
    if not tensors:
        return None
    x = torch.stack(tensors).cuda()
    feat = model(x)
    return F.normalize(feat, dim=-1).cpu()


def build_records(jsonl, topics):
    recs = defaultdict(list)
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("topic") in topics:
                recs[r["topic"]].append(r)
    return recs


def cos(a, b):
    return float((a * b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint tag, e.g. vigor_cross / cvusa")
    ap.add_argument("--repo_dir", default="/home/ain480/models_src/Sample4Geo")
    ap.add_argument("--dataset_dir", required=True,
                    help="DATASET_DIR; image base = <dataset_dir>/benchmark")
    ap.add_argument("--jsonl", default=None,
                    help="labeled jsonl; default <dataset_dir>/benchmark/benchmark_with_answers.jsonl")
    ap.add_argument("--outdir", default="results_sample4geo")
    ap.add_argument("--topics", nargs="*", default=MISMATCH_TOPICS)
    ap.add_argument("--limit", type=int, default=0, help="0=all; else first N per topic")
    ap.add_argument("--calib_frac", type=float, default=0.2,
                    help="fraction of binary Qs (stratified by gt) held out to pick threshold")
    ap.add_argument("--batch", type=int, default=64)
    args = parser_post(ap.parse_args())

    base = os.path.join(args.dataset_dir, "benchmark")
    jsonl = args.jsonl or os.path.join(base, "benchmark_with_answers.jsonl")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[load] {args.tag} <- {args.ckpt}")
    model = load_model(args.ckpt, args.repo_dir)

    recs = build_records(jsonl, args.topics)
    for t in args.topics:
        print(f"  {t}: {len(recs.get(t, []))} questions")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    raw = []   # per-question raw records (sims + pred + gt)

    # ---- score MCQ topics (argmax) ----
    for topic in [t for t in args.topics if "mcq" in t]:
        rows = recs.get(topic, [])
        if args.limit:
            rows = rows[:args.limit]
        for r in rows:
            sat_t = load_sat_tensor(os.path.join(base, r["images"]["satellite"]))
            sat_e = embed_batch(model, [sat_t])[0]
            opt_paths = r["option_stv_paths"]
            letters = sorted(opt_paths.keys())
            grds = [load_ground_tensor(angle_paths_for_sid(r["images"], base, opt_paths[L]))
                    for L in letters]
            grd_e = embed_batch(model, grds)
            sims = {L: cos(sat_e, grd_e[i]) for i, L in enumerate(letters)}
            pred = max(sims, key=sims.get)
            raw.append({"question_id": r["question_id"], "topic": topic,
                        "kind": "mcq", "gt": r["answer"], "pred": pred,
                        "correct": int(pred == r["answer"]), "sims": sims})

    # ---- score BINARY topics (threshold; calibrate on held-out slice) ----
    bin_topics = [t for t in args.topics if "binary" in t]
    bin_items = []  # (record, sim, gt_is_match)
    for topic in bin_topics:
        rows = recs.get(topic, [])
        if args.limit:
            rows = rows[:args.limit]
        for r in rows:
            sat_t = load_sat_tensor(os.path.join(base, r["images"]["satellite"]))
            sat_e = embed_batch(model, [sat_t])[0]
            gt_match = bool(r["mismatch_is_match"])
            if gt_match:
                ap_paths = angle_paths_for_sid(r["images"], base)         # own SV
            else:
                ap_paths = angle_paths_for_sid(r["images"], base,
                                               r["mismatch_negative_stv_paths"])  # negative SV
            grd_e = embed_batch(model, [load_ground_tensor(ap_paths)])[0]
            sim = cos(sat_e, grd_e)
            bin_items.append((r, topic, sim, gt_match))

    threshold = None
    if bin_items:
        threshold = calibrate_threshold(bin_items, args.calib_frac)
        print(f"[binary] calibrated threshold = {threshold:.4f} "
              f"(predict match if cosine > threshold)")
        for (r, topic, sim, gt_match) in bin_items:
            pred_match = sim > threshold
            # map to option letter: the option whose TEXT encodes the prediction
            pred_letter = letter_for_match(r["options"], pred_match)
            raw.append({"question_id": r["question_id"], "topic": topic,
                        "kind": "binary", "gt": r["answer"], "pred": pred_letter,
                        "correct": int(pred_letter == r["answer"]),
                        "sim": sim, "gt_is_match": gt_match,
                        "pred_is_match": bool(pred_match)})

    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**3)

    # ---- score ----
    per_topic = defaultdict(lambda: [0, 0])
    for x in raw:
        per_topic[x["topic"]][0] += x["correct"]
        per_topic[x["topic"]][1] += 1

    summary = {"tag": args.tag, "ckpt": args.ckpt, "threshold": threshold,
               "runtime_s": runtime, "peak_vram_gib": peak,
               "aggregation": "stitch_4angles_fwd_right_bwd_left->384x768",
               "n_total": len(raw), "per_topic": {}}
    print(f"\n=== {args.tag} ===")
    for t in args.topics:
        c, n = per_topic[t]
        acc = c / n if n else float("nan")
        summary["per_topic"][t] = {"n": n, "correct": c, "acc": acc}
        print(f"  {t:24s} n={n:4d} acc={acc:.4f}")
    overall_c = sum(v[0] for v in per_topic.values())
    overall_n = sum(v[1] for v in per_topic.values())
    summary["overall"] = {"n": overall_n, "correct": overall_c,
                          "acc": overall_c / overall_n if overall_n else float("nan")}
    print(f"  {'OVERALL':24s} n={overall_n:4d} acc={summary['overall']['acc']:.4f}")
    print(f"  runtime={runtime/60:.2f} min  peak_vram={peak:.2f} GiB  threshold={threshold}")

    with open(os.path.join(args.outdir, f"sample4geo_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, f"sample4geo_{args.tag}_raw.jsonl"), "w") as f:
        for x in raw:
            f.write(json.dumps(x) + "\n")
    print(f"  saved -> {args.outdir}/sample4geo_{args.tag}_*.json[l]")


def parser_post(a):
    return a


def letter_for_match(options, pred_match):
    """Return the option letter whose text means 'match' (if pred_match) else 'no match'."""
    for L, text in options.items():
        is_yes = text.strip().lower().startswith("yes")
        if is_yes == pred_match:
            return L
    # fallback: first letter
    return sorted(options.keys())[0]


def calibrate_threshold(bin_items, calib_frac):
    """Stratified held-out slice -> sweep candidate thresholds -> best balanced acc.
    Deterministic split (sorted by question_id, take every k-th into calib)."""
    items = sorted(bin_items, key=lambda x: x[0]["question_id"])
    pos = [x for x in items if x[3]]
    neg = [x for x in items if not x[3]]
    def take(group):
        k = max(1, int(round(1 / max(calib_frac, 1e-6))))
        return [g for i, g in enumerate(group) if i % k == 0]
    calib = take(pos) + take(neg)
    sims = sorted(set(round(x[2], 4) for x in calib))
    cands = sims + [(-1.0)] + [1.0]
    best_thr, best_score = 0.5, -1.0
    for thr in cands:
        tp = sum(1 for x in calib if x[3] and x[2] > thr)
        tn = sum(1 for x in calib if (not x[3]) and x[2] <= thr)
        np_ = sum(1 for x in calib if x[3]) or 1
        nn_ = sum(1 for x in calib if not x[3]) or 1
        bal = 0.5 * (tp / np_) + 0.5 * (tn / nn_)
        if bal > best_score:
            best_score, best_thr = bal, thr
    print(f"[binary] calib slice n={len(calib)} (pos={len(take(pos))} neg={len(take(neg))}) "
          f"best_balanced_acc={best_score:.4f}")
    return float(best_thr)


if __name__ == "__main__":
    main()
