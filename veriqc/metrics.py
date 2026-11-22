"""Task-appropriate evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "image_auroc",
    "pixel_auroc",
    "average_precision",
    "au_pro",
    "best_f1",
    "clopper_pearson",
]


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    masks = np.asarray(masks).astype(bool)
    if not masks.any():
        return float("nan")
    return float(roc_auc_score(masks.reshape(-1), np.asarray(maps, dtype=np.float64).reshape(-1)))


def average_precision(maps: np.ndarray, masks: np.ndarray) -> float:
    masks = np.asarray(masks).astype(bool)
    if not masks.any():
        return float("nan")
    return float(average_precision_score(masks.reshape(-1), np.asarray(maps, dtype=np.float64).reshape(-1)))


def au_pro(
    masks: np.ndarray,
    maps: np.ndarray,
    fpr_limit: float = 0.3,
    n_thresholds: int = 101,
) -> float:
    """Area under the per-region overlap curve, truncated at ``fpr_limit``."""

    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float32)
    normal_pixels = [maps[i][~masks[i]].reshape(-1) for i in range(len(masks))]
    normal_pixels = [p for p in normal_pixels if p.size]
    if not normal_pixels:
        return float("nan")
    normal = np.concatenate(normal_pixels)
    thresholds = np.quantile(normal, np.linspace(1.0, 0.0, n_thresholds))
    regions: list[np.ndarray] = []
    for index in range(len(masks)):
        labels, count = ndimage.label(masks[index])
        for region_id in range(1, count + 1):
            values = maps[index][labels == region_id]
            if values.size:
                regions.append(np.sort(values))
    if not regions:
        return float("nan")
    pro = []
    fprs = []
    for threshold in thresholds:
        covered = [float((values >= threshold).mean()) for values in regions]
        pro.append(float(np.mean(covered)))
        fprs.append(float((normal >= threshold).mean()))
    pro_array = np.asarray(pro)
    fpr_array = np.asarray(fprs)
    keep = fpr_array <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    return float(np.trapezoid(pro_array[keep], fpr_array[keep]) / fpr_limit)


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(labels.sum(), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1))
    return float(f1[index]), float(precision[index]), float(recall[index])


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n <= 0:
        return float("nan"), float("nan")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper
