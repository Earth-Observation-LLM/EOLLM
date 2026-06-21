"""metrics.py — accuracy / F1 / ROC-AUC / CIs + rich slices.

Vendored verbatim from benchmark_modes_remote/metrics.py, with scikit-learn made
OPTIONAL: accuracy, slices, bootstrap CI and answer distributions need only
numpy and always run; F1 / precision / recall / ROC-AUC require scikit-learn and
are skipped gracefully (with a recorded note) when it is absent, so the suite
runs on the bare `unsloth` env. Install scikit-learn to enable them.
"""
from collections import defaultdict, Counter
import random
import numpy as np

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


LABELS = ["A", "B", "C", "D"]
ALL_VALID = set(LABELS)


def _row_labels(row):
    """Return the actual answer letters present in this row's options dict.
    Falls back to full LABELS for rows that pre-date the options field."""
    opts = row.get("options")
    if opts and isinstance(opts, dict):
        return sorted(k for k in opts if k in ALL_VALID)
    return LABELS


def accuracy(rows):
    valid = [r for r in rows if r.get("gold") in ALL_VALID]
    if not valid:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    correct = sum(1 for r in valid if r["gold"] == r["prediction"])
    return {"accuracy": correct / len(valid), "correct": correct, "total": len(valid)}


def slice_accuracy(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(key, "unknown")].append(r)
    return {
        str(k): accuracy(v)
        for k, v in sorted(buckets.items(), key=lambda x: str(x[0]))
    }


def interaction_accuracy(rows, key1, key2):
    buckets = defaultdict(list)
    for r in rows:
        v1 = r.get(key1, "unknown")
        v2 = r.get(key2, "unknown")
        buckets[(v1, v2)].append(r)
    return {
        f"{key1}={a} | {key2}={b}": accuracy(v)
        for (a, b), v in sorted(buckets.items(), key=lambda x: str(x[0]))
    }

def answer_distribution(rows):
    return {
        "gold":       dict(Counter(r.get("gold")       for r in rows)),
        "prediction": dict(Counter(r.get("prediction") for r in rows)),
    }


def bootstrap_ci(rows, n_samples=1000, seed=42):
    random.seed(seed)
    if not rows:
        return {"mean": 0.0, "p05": 0.0, "p95": 0.0}
    accs = sorted(
        accuracy([random.choice(rows) for _ in rows])["accuracy"]
        for _ in range(n_samples)
    )
    return {
        "mean": sum(accs) / len(accs),
        "p05":  accs[int(0.05 * len(accs))],
        "p95":  accs[int(0.95 * len(accs))],
    }


def classification_metrics(rows):
    """
    F1 / Precision / Recall computed per-row against only the labels that
    actually exist for that question (A/B for binary tasks, A/B/C/D for the rest).
    Rows are grouped by their label-set so sklearn sees a consistent label space
    within each group, then results are pooled with weighted averaging.
    """
    if not _HAVE_SKLEARN:
        return {"skipped": "scikit-learn not installed"}
    # Bucket rows by their label-set tuple, e.g. ('A','B') vs ('A','B','C','D')
    buckets = defaultdict(list)
    for r in rows:
        labels = tuple(_row_labels(r))
        valid_labels = set(labels)
        if r.get("gold") in valid_labels and r.get("prediction") in valid_labels:
            buckets[labels].append(r)

    if not buckets:
        return {"error": "No rows with valid gold and prediction labels."}

    # Collect per-bucket metrics then pool
    all_true, all_pred, all_labels_per_row = [], [], []
    per_labelset = {}

    for label_tuple, bucket_rows in sorted(buckets.items()):
        y_true = [r["gold"]       for r in bucket_rows]
        y_pred = [r["prediction"] for r in bucket_rows]
        labels = list(label_tuple)

        per_labelset["/".join(labels)] = {
            "n":            len(bucket_rows),
            "weighted_f1":  round(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 4),
            "macro_f1":     round(f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0), 4),
        }

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_labels_per_row.extend([labels] * len(bucket_rows))

    # Overall: use all observed labels so the pooled F1 is fair
    observed_labels = sorted(set(all_true) | set(all_pred))

    return {
        "n_clf":              len(all_true),
        "weighted_f1":        round(f1_score(all_true, all_pred, average="weighted", labels=observed_labels, zero_division=0), 4),
        "macro_f1":           round(f1_score(all_true, all_pred, average="macro",    labels=observed_labels, zero_division=0), 4),
        "weighted_precision": round(precision_score(all_true, all_pred, average="weighted", labels=observed_labels, zero_division=0), 4),
        "weighted_recall":    round(recall_score(   all_true, all_pred, average="weighted", labels=observed_labels, zero_division=0), 4),
        "per_labelset":       per_labelset,
    }


def roc_curve_for_topic(rows):
    """
    Macro-averaged ROC curve for a topic slice.
    Handles mixed binary (A/B) and 4-way (A/B/C/D) rows within the same topic
    by computing OVR curves only against the labels that exist per row.
    """
    if not _HAVE_SKLEARN:
        return {"skipped": "scikit-learn not installed"}
    auc_rows = [
        r for r in rows
        if r.get("gold") in ALL_VALID
        and r.get("prediction") in ALL_VALID
        and r.get("prob_dict")
    ]

    if len(auc_rows) < 2:
        return {"error": "Not enough rows with prob_dict."}

    # Determine the effective label set for this topic slice
    observed = sorted(set(r["gold"] for r in auc_rows) | set(r["prediction"] for r in auc_rows))
    if len(observed) < 2:
        return {"error": "Not enough class diversity (needs at least 2 different gold labels)."}

    y_true = [r["gold"] for r in auc_rows]
    probs  = [r["prob_dict"] for r in auc_rows]

    # Build score matrix using only observed labels; fill missing keys with 0
    y_true_bin = label_binarize(y_true, classes=observed)
    if y_true_bin.shape[1] == 1:
        # label_binarize collapses to 1 column for binary — expand manually
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    y_scores = np.array([[p.get(lbl, 0.0) for lbl in observed] for p in probs])

    # Renormalise so extracted token probs sum to 1 per row
    row_sums = y_scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_scores = y_scores / row_sums

    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []

    for i, label in enumerate(observed):
        positives = y_true_bin[:, i].sum()
        if positives == 0 or positives == len(y_true_bin):
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc_score(y_true_bin[:, i], y_scores[:, i]))

    if not tprs:
        return {"error": "No valid per-label curves could be computed."}

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    result = {
        "n":          len(auc_rows),
        "labels":     observed,
        "roc_auc":    round(float(np.mean(aucs)), 4),
        "fpr":        np.nan_to_num(mean_fpr).tolist(),
        "tpr":        np.nan_to_num(mean_tpr).tolist(),
    }

    if len(auc_rows) < 20:
        result["warning"] = f"Only {len(auc_rows)} samples — AUC may be unreliable."

    return result


def roc_curves_by_topic(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("topic", "unknown")].append(r)
    return {
        topic: roc_curve_for_topic(bucket_rows)
        for topic, bucket_rows in sorted(buckets.items())
    }


def build_full_report(rows):
    return {
        "overall":                accuracy(rows),
        "classification_metrics": classification_metrics(rows),
        "confidence_interval":    bootstrap_ci(rows),
        "answer_distribution":    answer_distribution(rows),

        "by_topic":               slice_accuracy(rows, "topic"),
        "by_difficulty":          slice_accuracy(rows, "difficulty"),
        "by_land_use":            slice_accuracy(rows, "land_use"),
        "by_country":              slice_accuracy(rows, "country"),
        "by_city":                slice_accuracy(rows, "city"),
        "by_city_type":           slice_accuracy(rows, "city_type"),
        "by_benchmark_city_type": slice_accuracy(rows, "benchmark_city_type"),
        "by_generation_method":   slice_accuracy(rows, "generation_method"),
        "by_mismatch_strategy":   slice_accuracy(rows, "mismatch_strategy"),
        "by_query_stv_angle":     slice_accuracy(rows, "query_stv_angle"),

        "interaction_topic_city_type":        interaction_accuracy(rows, "topic",       "city_type"),
        "interaction_difficulty_city_type":   interaction_accuracy(rows, "difficulty",  "city_type"),
        "interaction_land_use_difficulty":    interaction_accuracy(rows, "land_use",    "difficulty"),
        "interaction_topic_difficulty":       interaction_accuracy(rows, "topic",       "difficulty"),

        "ablation_mode":      rows[0].get("ablation_mode") if rows else None,
    }