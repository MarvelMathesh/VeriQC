"""Anomaly-detection metrics: image/pixel AUROC, AU-PRO, FPR and miss rate.

AU-PRO (Perlin et al., NeurIPS 2021) is the segmentation quality metric used
by the MVTec AD benchmark and by every patch-memory method we compare against.
It is implemented here from the definition rather than copied, because a
silently wrong AU-PRO would corrupt every conclusion in the report.

Definition used (the canonical one)
-----------------------------------
* A *region* is a connected component of the binary ground-truth mask.
* For a threshold t, the per-region detection fraction is
  ``covered pixels in region / region area``.
* ``PRO(t)`` is the **mean over regions** of that fraction (regions are
  weighted equally, so a 5-pixel defect counts as much as a 5000-pixel one;
  that is the whole point of the metric).
* ``FPR(t)`` is the fraction of *normal* pixels (from normal test images)
  that the threshold marks as anomalous.
* ``AU-PRO`` is the area under the ``PRO`` vs ``FPR`` curve, truncated at
  ``fpr_limit`` and normalised by it, so 0.5 is the chance level and the
  value reported by the MVTec leaderboard style ("AU-PRO @ 0.3 FPR").

Thresholds are placed at quantiles of the normal-pixel score distribution, so
the FPR grid is exact rather than an artefact of an arbitrary score grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

__all__ = [
    "AnomalyScores",
    "auroc_image",
    "auroc_pixel",
    "compute_au_pro",
    "fpr_miss_at_threshold",
    "normalise_maps",
]


@dataclass
class AnomalyScores:
    """Scores for one dataset split.

    image_scores : (N,) per-image anomaly score, higher = more anomalous
    masks        : (N, H, W) bool ground truth, False on normal images
    maps         : (N, H, W) float pixel anomaly maps, higher = more anomalous
    """

    image_scores: np.ndarray
    masks: np.ndarray
    maps: np.ndarray

    def __post_init__(self) -> None:
        self.image_scores = np.asarray(self.image_scores, dtype=np.float64).reshape(-1)
        self.masks = np.asarray(self.masks).astype(bool)
        self.maps = np.asarray(self.maps, dtype=np.float32)
        n = len(self.image_scores)
        if len(self.masks) != n or len(self.maps) != n:
            raise ValueError("scores/masks/maps disagree in length")


def normalise_maps(maps: np.ndarray) -> np.ndarray:
    """Global min-max normalisation of the anomaly maps over the whole split.

    Global (not per image) so that map magnitudes stay comparable across
    images, which is what the threshold sweep assumes.
    """
    maps = np.asarray(maps, dtype=np.float32)
    lo = float(np.nanmin(maps))
    hi = float(np.nanmax(maps))
    if hi - lo < 1e-12:
        return np.zeros_like(maps)
    return (maps - lo) / (hi - lo)


def auroc_image(image_scores: np.ndarray, is_anomalous: np.ndarray) -> float:
    """Image-level AUROC."""
    y = np.asarray(is_anomalous).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(image_scores, dtype=np.float64)))


def auroc_pixel(maps: np.ndarray, masks: np.ndarray) -> float:
    """Pixel-level AUROC over all pixels of all images."""
    m = np.asarray(masks).astype(bool).reshape(-1)
    s = np.asarray(maps, dtype=np.float64).reshape(-1)
    if m.min() == m.max():
        return float("nan")
    return float(roc_auc_score(m, s))


def _regions(masks: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per anomalous image: (region_labels, region_count)."""
    out = []
    for m in masks:
        lab, n = ndimage.label(m)
        out.append((lab, n))
    return out


def compute_au_pro(
    masks: np.ndarray,
    maps: np.ndarray,
    fpr_limit: float = 0.3,
    n_fpr: int = 301,
    region_weighted: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """AU-PRO at a given FPR limit.

    Parameters
    ----------
    masks : (N, H, W) bool ground truth
    maps  : (N, H, W) float anomaly maps (any monotone scale)
    fpr_limit : truncate the curve here (0.3 is the MVTec convention)
    n_fpr : number of FPR points on [0, 1]
    region_weighted : if True, weight regions by area instead of treating every
        region equally. The canonical metric does *not* weight by area; the
        flag exists so we can report the sensitivity.

    Returns
    -------
    (au_pro, fprs, pro_curve)
    """
    masks = np.asarray(masks).astype(bool)
    maps = normalise_maps(maps)
    n = len(masks)

    normal_idx = [i for i in range(n) if not masks[i].any()]
    anom_idx = [i for i in range(n) if masks[i].any()]

    # ---- threshold grid from normal pixel scores -------------------------
    if normal_idx:
        normal_scores = np.concatenate([maps[i].reshape(-1) for i in normal_idx])
        normal_scores.sort()
        n_normal = len(normal_scores)
    else:
        normal_scores = np.zeros(1, dtype=np.float32)
        n_normal = 0

    fprs = np.linspace(0.0, 1.0, n_fpr)
    # threshold that lets a fraction f of normal pixels through
    thresholds = np.empty(n_fpr, dtype=np.float32)
    for k, f in enumerate(fprs):
        # we want: fraction of normal pixels with score > t  ==  f
        # normal_scores sorted ascending; index = n*(1-f) from the top
        q = min(max(int(round((1.0 - f) * n_normal)), 0), max(n_normal - 1, 0))
        thresholds[k] = normal_scores[q] if n_normal else 1.0

    # ---- per-region score sorting for fast coverage counting ------------
    # For every region we sort its pixel scores once; the covered fraction at
    # any threshold is then a searchsorted lookup.
    region_sorted: list[np.ndarray] = []
    region_areas: list[int] = []
    for i in anom_idx:
        lab, n_reg = ndimage.label(masks[i])
        for r in range(1, n_reg + 1):
            sel = lab == r
            region_sorted.append(np.sort(maps[i][sel]))
            region_areas.append(int(sel.sum()))

    total_regions = len(region_sorted)
    pro = np.zeros(n_fpr, dtype=np.float64)
    if total_regions == 0:
        return float("nan"), fprs, pro

    # count covered pixels per region per threshold
    covered = np.empty((n_fpr, total_regions), dtype=np.int32)
    for ri, scores in enumerate(region_sorted):
        # number of pixels strictly greater than each threshold
        pos = len(scores) - np.searchsorted(scores, thresholds, side="right")
        covered[:, ri] = pos

    fracs = covered / np.maximum(np.array(region_areas)[None, :], 1)
    if region_weighted:
        w = np.array(region_areas, dtype=np.float64)
        pro = (fracs * w[None, :]).sum(axis=1) / w.sum()
    else:
        pro = fracs.mean(axis=1)

    # ---- integrate -------------------------------------------------------
    keep = fprs <= fpr_limit + 1e-12
    if keep.sum() < 2:
        return float("nan"), fprs, pro
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    au = float(trapz(pro[keep], fprs[keep]) / fpr_limit)
    return au, fprs, pro


def fpr_miss_at_threshold(
    image_scores: np.ndarray,
    is_anomalous: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """False-alarm rate on normals and miss rate on defects at one threshold."""
    s = np.asarray(image_scores, dtype=np.float64)
    y = np.asarray(is_anomalous).astype(bool)
    normals = ~y
    anomalies = y
    fpr = float((s[normals] > threshold).mean()) if normals.any() else float("nan")
    miss = float((s[anomalies] <= threshold).mean()) if anomalies.any() else float("nan")
    return {"fpr": fpr, "miss_rate": miss, "threshold": float(threshold)}
