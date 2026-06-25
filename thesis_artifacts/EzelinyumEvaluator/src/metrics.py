"""Metrics — Wilson CI, per-topic chance/majority baselines, per-topic F1/AUC
with PER-RECORD option space (audit fix), bootstrap CIs.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------

def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a Bernoulli proportion.

    Far better than normal-approximation for small N or extreme proportions.
    """
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 3407,
) -> tuple[float, float, float]:
    """Returns (mean, low, high) of a percentile bootstrap CI."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(values)
    arr = np.asarray(values, dtype=float)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    return (float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Per-topic chance + majority baselines
# ---------------------------------------------------------------------------

def per_topic_baselines(
    records: list[dict],
) -> dict[str, dict]:
    """For each topic, compute:
    - n_options (avg)
    - random_baseline (1 / n_options, weighted by sample count)
    - majority_baseline (most-frequent ground-truth letter's share)
    """
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r)

    out = {}
    for topic, rows in by_topic.items():
        option_sizes = [len(r["options"]) for r in rows]
        avg_opts = sum(option_sizes) / len(option_sizes)
        random_baseline = sum(1.0 / s for s in option_sizes) / len(option_sizes)
        golds = Counter(r["answer"] for r in rows)
        majority_baseline = max(golds.values()) / len(rows)
        out[topic] = {
            "n": len(rows),
            "avg_options": avg_opts,
            "random_baseline": random_baseline,
            "majority_baseline": majority_baseline,
            "gold_distribution": dict(golds),
        }
    return out


# ---------------------------------------------------------------------------
# Per-topic F1 and AUC — with PER-RECORD option space (the audit fix)
# ---------------------------------------------------------------------------

def per_topic_metrics(
    rows: list[dict],
) -> dict:
    """For a list of prediction rows (one topic), compute:
    - accuracy, Wilson CI, n correct, n total
    - macro/micro/weighted F1 over THE OPTIONS THAT ACTUALLY EXIST
    - per-label precision/recall/F1
    - macro-AUC over the same restricted label set (if prob_dict present)

    Each row must have:
        gold (str), pred (Optional[str]), correct (bool), options (dict),
        prob_dict (Optional[dict[str, float]])
    """
    if not rows:
        return {"n": 0}

    # Determine the label set ACROSS this topic (union of options seen).
    label_set: set[str] = set()
    for r in rows:
        label_set.update(r["options"].keys())
    labels = sorted(label_set)

    # Accuracy & Wilson CI.
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / n
    lo, hi = wilson_ci(correct, n)

    # Precision/recall/F1 per label.
    tp = {L: 0 for L in labels}
    fp = {L: 0 for L in labels}
    fn = {L: 0 for L in labels}
    for r in rows:
        g = r["gold"]
        p = r["pred"]
        if p == g:
            tp[g] += 1
        else:
            if p in labels:
                fp[p] += 1
            fn[g] += 1

    per_label = {}
    for L in labels:
        prec = tp[L] / (tp[L] + fp[L]) if (tp[L] + fp[L]) else 0.0
        rec = tp[L] / (tp[L] + fn[L]) if (tp[L] + fn[L]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for r in rows if r["gold"] == L)
        per_label[L] = dict(precision=prec, recall=rec, f1=f1, support=support)

    # Macro F1: mean over labels that have non-zero support.
    supported = [L for L in labels if per_label[L]["support"] > 0]
    macro_f1 = sum(per_label[L]["f1"] for L in supported) / max(len(supported), 1)
    weighted_f1 = (
        sum(per_label[L]["f1"] * per_label[L]["support"] for L in supported)
        / sum(per_label[L]["support"] for L in supported)
    ) if supported else 0.0
    # Micro F1 == accuracy for single-label problems.
    micro_f1 = acc

    # Per-label AUC if prob_dict present. Restricted to label_set.
    aucs = {}
    if any(r.get("prob_dict") for r in rows):
        for L in labels:
            y_true = [1 if r["gold"] == L else 0 for r in rows]
            y_score = []
            for r in rows:
                pd = r.get("prob_dict") or {}
                # If pd has only some labels, missing -> 0.
                y_score.append(float(pd.get(L, 0.0)))
            positives = sum(y_true)
            if 0 < positives < len(y_true):
                aucs[L] = _binary_auc(y_true, y_score)
        macro_auc = sum(aucs.values()) / len(aucs) if aucs else None
    else:
        macro_auc = None

    # Hedging / refusal rates.
    n_hedged = sum(1 for r in rows if r.get("hedged"))
    n_refused = sum(1 for r in rows if r.get("refused"))
    n_unparseable = sum(1 for r in rows if r["pred"] is None and not r.get("refused"))

    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "ci95_low": lo,
        "ci95_high": hi,
        "labels": labels,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "per_label_auc": aucs,
        "macro_auc": macro_auc,
        "n_hedged": n_hedged,
        "n_refused": n_refused,
        "n_unparseable": n_unparseable,
    }


def _binary_auc(y_true: list[int], y_score: list[float]) -> float:
    """Mann-Whitney U formulation of AUC (no sklearn dependency)."""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n1 = sum(y_true)
    n0 = len(y_true) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    # Rank-sum
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) - 1 and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum_pos - n1 * (n1 + 1) / 2) / (n0 * n1)
