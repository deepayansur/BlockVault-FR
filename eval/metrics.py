from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

EPS = 1e-12


def _prepare_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores_arr.size != labels_arr.size:
        raise ValueError("scores and labels must have the same length")
    if scores_arr.size == 0:
        raise ValueError("scores and labels must be non-empty")

    labels_arr = (labels_arr > 0).astype(np.int32)
    positives = int(labels_arr.sum())
    negatives = int(labels_arr.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("scores must include both positive and negative labels")
    return scores_arr, labels_arr


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_arr, labels_arr = _prepare_inputs(scores, labels)
    order = np.argsort(-scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    sorted_labels = labels_arr[order]

    distinct_end = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_idx = np.r_[distinct_end, sorted_scores.size - 1]

    true_positives = np.cumsum(sorted_labels, dtype=np.float64)[threshold_idx]
    false_positives = (threshold_idx + 1).astype(np.float64) - true_positives

    positive_total = float(sorted_labels.sum())
    negative_total = float(sorted_labels.size - sorted_labels.sum())

    thresholds = sorted_scores[threshold_idx]
    tprs = true_positives / max(positive_total, EPS)
    fprs = false_positives / max(negative_total, EPS)

    thresholds = np.r_[np.inf, thresholds]
    tprs = np.r_[0.0, tprs]
    fprs = np.r_[0.0, fprs]
    return thresholds.astype(np.float64), fprs.astype(np.float64), tprs.astype(np.float64)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    _, fprs, tprs = roc_curve(scores, labels)
    return float(np.trapezoid(tprs, fprs))


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds, fprs, tprs = roc_curve(scores, labels)
    fnrs = 1.0 - tprs
    diff = fprs - fnrs

    for idx in range(1, diff.size):
        prev = diff[idx - 1]
        cur = diff[idx]
        if cur == 0:
            return float(fprs[idx]), float(thresholds[idx])
        if prev == 0:
            return float(fprs[idx - 1]), float(thresholds[idx - 1])
        if (prev < 0 < cur) or (prev > 0 > cur):
            weight = abs(prev) / (abs(prev) + abs(cur))
            eer_value = fprs[idx - 1] + weight * (fprs[idx] - fprs[idx - 1])
            if np.isfinite(thresholds[idx - 1]) and np.isfinite(thresholds[idx]):
                threshold = thresholds[idx - 1] + weight * (thresholds[idx] - thresholds[idx - 1])
            else:
                threshold = thresholds[idx]
            return float(eer_value), float(threshold)

    best_idx = int(np.argmin(np.abs(diff)))
    eer_value = (fprs[best_idx] + fnrs[best_idx]) / 2.0
    return float(eer_value), float(thresholds[best_idx])


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float) -> tuple[float, float]:
    thresholds, fprs, tprs = roc_curve(scores, labels)
    valid = np.where(fprs <= far)[0]
    if valid.size == 0:
        return 0.0, float(thresholds[0])

    best_tpr = float(np.max(tprs[valid]))
    candidates = valid[np.where(np.isclose(tprs[valid], best_tpr))[0]]
    best_idx = int(candidates[np.argmax(thresholds[candidates])])
    return float(tprs[best_idx]), float(thresholds[best_idx])


def fnmr_at_fmr(scores: np.ndarray, labels: np.ndarray, fmr: float) -> tuple[float, float]:
    tar, threshold = tar_at_far(scores, labels, fmr)
    return float(1.0 - tar), float(threshold)


def far_summaries(scores: np.ndarray, labels: np.ndarray, fars: Iterable[float]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for far in fars:
        tar, threshold = tar_at_far(scores, labels, far)
        summary[f"{far:.0e}"] = {
            "far": float(far),
            "tar": float(tar),
            "threshold": float(threshold),
            "fnmr": float(1.0 - tar),
        }
    return summary


def bootstrap_verification(
    scores: np.ndarray,
    labels: np.ndarray,
    fars: Iterable[float],
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    if bootstrap_samples <= 0:
        return {}

    scores_arr, labels_arr = _prepare_inputs(scores, labels)
    rng = np.random.default_rng(seed)
    far_values = tuple(fars)
    metrics: Dict[str, list[float]] = {"eer": [], "roc_auc": []}
    for far in far_values:
        metrics[f"tar@{far:.0e}"] = []

    for _ in range(bootstrap_samples):
        sample_idx = rng.integers(0, scores_arr.size, size=scores_arr.size)
        sample_scores = scores_arr[sample_idx]
        sample_labels = labels_arr[sample_idx]
        if np.unique(sample_labels).size < 2:
            continue

        eer_value, _ = eer(sample_scores, sample_labels)
        metrics["eer"].append(eer_value)
        metrics["roc_auc"].append(roc_auc(sample_scores, sample_labels))

        for far in far_values:
            tar_value, _ = tar_at_far(sample_scores, sample_labels, far)
            metrics[f"tar@{far:.0e}"] .append(tar_value)

    intervals: Dict[str, Dict[str, float]] = {}
    for key, values in metrics.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        intervals[key] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
            "samples": int(arr.size),
        }
    return intervals


def summarize_verification(
    scores: np.ndarray,
    labels: np.ndarray,
    fars: Iterable[float] = (1e-2, 1e-3, 1e-4),
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> Dict[str, object]:
    scores_arr, labels_arr = _prepare_inputs(scores, labels)
    far_values = tuple(fars)
    eer_value, eer_threshold = eer(scores_arr, labels_arr)
    return {
        "pairs": int(scores_arr.size),
        "positives": int(labels_arr.sum()),
        "negatives": int(labels_arr.size - labels_arr.sum()),
        "eer": {
            "value": float(eer_value),
            "threshold": float(eer_threshold),
        },
        "roc_auc": float(roc_auc(scores_arr, labels_arr)),
        "far_targets": far_summaries(scores_arr, labels_arr, far_values),
        "confidence_intervals": bootstrap_verification(
            scores_arr,
            labels_arr,
            fars=far_values,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }
