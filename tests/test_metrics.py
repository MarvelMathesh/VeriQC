"""Unit tests for the metric implementations (E0 gate).

These are the tests that decide whether every later number in the report can
be trusted, so they check the metrics against hand-computed values rather than
against another library that might share the same misreading.
"""

from __future__ import annotations

import numpy as np
import pytest

from veriqc.metrics.anomaly import (
    auroc_image,
    auroc_pixel,
    compute_au_pro,
    fpr_miss_at_threshold,
    normalise_maps,
)
from veriqc.metrics.conformal import (
    conformal_threshold,
    cp_lower_fpr,
    cp_upper_fpr,
    min_n_for_target,
)
from veriqc.metrics.detection import (
    Detection,
    average_precision,
    confusion_matrix,
    detection_metrics,
    mean_average_precision,
)


# --------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------
def test_auroc_perfect_and_chance():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert auroc_image(s, y) == 1.0
    # all-normal or all-anomalous is undefined, not 0.5
    assert np.isnan(auroc_image(s, np.zeros(4, int)))


def test_auroc_matches_sklearn_on_random_data():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = (rng.random(500) < 0.3).astype(int)
    assert auroc_image(s, y) == pytest.approx(roc_auc_score(y, s))


def test_auroc_pixel():
    maps = np.array([[0.1, 0.9], [0.2, 0.8]])
    masks = np.array([[0, 1], [0, 1]], dtype=bool)
    assert auroc_pixel(maps, masks) == 1.0


# --------------------------------------------------------------------------
# AU-PRO
# --------------------------------------------------------------------------
def test_aupro_perfect_single_region():
    """One region, perfectly separated from the normals -> AU-PRO 1.0."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 0, 0] = True
    maps = np.zeros((2, 4, 4), dtype=np.float32)
    maps[1, 0, 0] = 1.0
    au, fprs, pro = compute_au_pro(masks, maps, fpr_limit=1.0, n_fpr=11)
    assert au == pytest.approx(1.0, abs=1e-6)
    assert np.all(pro >= 0.99)


def test_aupro_zero_when_region_invisible():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 0, 0] = True
    maps = np.zeros((2, 4, 4), dtype=np.float32)  # anomaly scores identical to normal
    au, _, _ = compute_au_pro(masks, maps, fpr_limit=1.0, n_fpr=11)
    assert au == pytest.approx(0.0, abs=1e-6)


def test_aupro_region_weighting_flag_differs():
    """Unweighted (canonical) counts a 1px region as much as a 9px region."""
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[1, 0, 0] = True  # 1 pixel region, detected
    masks[1, 2:5, 2:5] = True  # 9 pixel region, not detected
    maps = np.zeros((2, 5, 5), dtype=np.float32)
    maps[1, 0, 0] = 1.0
    au_unweighted, _, pro_u = compute_au_pro(
        masks, maps, fpr_limit=1.0, n_fpr=11, region_weighted=False
    )
    au_weighted, _, pro_w = compute_au_pro(
        masks, maps, fpr_limit=1.0, n_fpr=11, region_weighted=True
    )
    assert pro_u[0] == pytest.approx(0.5, abs=1e-6)  # (1.0 + 0.0) / 2
    assert pro_w[0] == pytest.approx(0.1, abs=1e-6)  # (1*1 + 0*9) / 10
    assert au_unweighted > au_weighted


def test_aupro_monotone_in_fpr_and_truncates_at_limit():
    rng = np.random.default_rng(3)
    masks = np.zeros((20, 16, 16), dtype=bool)
    masks[10:, 4:8, 4:8] = True
    maps = rng.random((20, 16, 16)).astype(np.float32)
    maps[10:, 4:8, 4:8] += 0.8  # anomalies score higher
    au_full, fprs, pro = compute_au_pro(masks, maps, fpr_limit=1.0, n_fpr=51)
    au_03, _, _ = compute_au_pro(masks, maps, fpr_limit=0.3, n_fpr=51)
    assert np.all(np.diff(pro) >= -1e-9)  # PRO is non-decreasing in FPR
    # the truncated *unnormalised* area cannot exceed the full one, while the
    # normalised value may be either side depending on where PRO saturates
    trapz = getattr(np, "trapezoid", None) or np.trapz
    keep = fprs <= 0.3 + 1e-12
    assert trapz(pro[keep], fprs[keep]) <= trapz(pro, fprs) + 1e-9
    assert 0.0 <= au_03 <= 1.0
    assert 0.0 <= au_full <= 1.0


def test_aupro_hand_computed_two_regions():
    """PRO at the tightest threshold: one of two regions detected -> 0.5."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 0, 0] = True
    masks[1, 1, 1] = True  # two separate 1px regions
    maps = np.zeros((2, 4, 4), dtype=np.float32)
    maps[1, 0, 0] = 1.0  # only the first region is visible
    _, _, pro = compute_au_pro(masks, maps, fpr_limit=1.0, n_fpr=11)
    assert pro[0] == pytest.approx(0.5, abs=1e-6)


def test_normalise_maps_is_global():
    maps = np.array([[[0.0, 5.0]], [[10.0, 2.0]]])
    out = normalise_maps(maps)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[0, 0, 1] == pytest.approx(0.5)  # global, not per-image


def test_fpr_miss_at_threshold():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    out = fpr_miss_at_threshold(s, y, 0.5)
    assert out["fpr"] == 0.0
    assert out["miss_rate"] == 0.0
    out2 = fpr_miss_at_threshold(s, y, 0.15)
    assert out2["fpr"] == 0.5  # 0.2 > 0.15 on one of two normals
    assert out2["miss_rate"] == 0.0  # both anomalies are above the threshold
    out3 = fpr_miss_at_threshold(s, y, 0.85)
    assert out3["miss_rate"] == 0.5  # 0.8 <= 0.85, 0.9 > 0.85


# --------------------------------------------------------------------------
# Conformal
# --------------------------------------------------------------------------
def test_conformal_not_achievable_when_n_too_small():
    rec = conformal_threshold(np.arange(10.0), alpha=0.01)
    assert not rec.achievable
    assert rec.threshold == float("inf")
    assert "n>=99" in rec.note  # alpha=0.01 needs n >= 1/alpha - 1
    rec_ok = conformal_threshold(np.arange(300.0), alpha=0.01)
    assert rec_ok.achievable
    # k = ceil(301 * 0.99) = 298 -> the 298th smallest of 0..299
    assert rec_ok.order_statistic_k == 298
    assert rec_ok.threshold == pytest.approx(297.0)


def test_conformal_quantile_is_the_ceil_order_statistic():
    s = np.arange(1.0, 101.0)  # 1..100
    rec = conformal_threshold(s, alpha=0.05)
    # k = ceil(101*0.95) = 96 -> the 96th smallest = 96.0
    assert rec.order_statistic_k == 96
    assert rec.threshold == pytest.approx(96.0)
    rec2 = conformal_threshold(s, alpha=0.01)
    # k = ceil(101*0.99) = 100 -> the 100th smallest = 100.0
    assert rec2.order_statistic_k == 100
    assert rec2.threshold == pytest.approx(100.0)


def test_conformal_empirical_fpr_is_near_alpha():
    rng = np.random.default_rng(7)
    cal = rng.normal(size=500)
    rec = conformal_threshold(cal, alpha=0.05)
    fresh = rng.normal(size=20000)
    fpr = float((fresh > rec.threshold).mean())
    assert 0.03 < fpr < 0.07


def test_clopper_pearson_matches_published_bound():
    """arXiv 2608.15090: a 1% target needs >= 299 normals at 95% confidence."""
    assert min_n_for_target(0.01) == 299
    assert min_n_for_target(0.05) == 59
    # with 150 normals and zero false alarms the 95% upper bound is ~1.98%
    assert cp_upper_fpr(150, 0) == pytest.approx(0.01973, abs=1e-4)
    assert cp_upper_fpr(299, 0) < 0.01
    assert cp_upper_fpr(150, 0) >= 0.01


# --------------------------------------------------------------------------
# Detection metrics
# --------------------------------------------------------------------------
def _det(pred, scores, labels, gt, gtl):
    return Detection(
        np.array(pred, dtype=float).reshape(-1, 4),
        np.array(scores, dtype=float),
        np.array(labels, dtype=int),
        np.array(gt, dtype=float).reshape(-1, 4),
        np.array(gtl, dtype=int),
    )


def test_ap_perfect_and_half():
    # one gt, one exact prediction -> AP 1
    tp = np.array([True])
    fp = np.array([False])
    assert average_precision(tp, fp, 1) == pytest.approx(1.0)
    # FP ranked above TP -> VOC AP = precision at recall 1 = 0.5
    tp = np.array([False, True])
    fp = np.array([True, False])
    assert average_precision(tp, fp, 1) == pytest.approx(0.5)


def test_map_perfect():
    dets = [
        _det([[10, 10, 50, 50]], [0.9], [0], [[10, 10, 50, 50]], [0]),
        _det([[0, 0, 20, 20]], [0.8], [1], [[0, 0, 20, 20]], [1]),
    ]
    res = mean_average_precision(dets, [0, 1], 0.5)
    assert res["mAP"] == pytest.approx(1.0)


def test_map_half_when_duplicate_prediction_is_first():
    """A duplicate FP ranked below the TP does not lower VOC-style AP.

    VOC interpolation takes, at each recall, the best precision achieved at
    that recall or higher, so the AP stays 1.0. The duplicate *is* penalised
    by the fixed-confidence precision/recall numbers, which is why both are
    reported.
    """
    dets = [
        _det(
            [[10, 10, 50, 50], [10, 10, 50, 50]],
            [0.9, 0.8],
            [0, 0],
            [[10, 10, 50, 50]],
            [0],
        )
    ]
    res = mean_average_precision(dets, [0], 0.5)
    assert res["mAP"] == pytest.approx(1.0)
    full = detection_metrics(dets, [0])
    assert full["precision@0.25"] == pytest.approx(0.5)  # the duplicate is an FP
    assert full["recall@0.25"] == pytest.approx(1.0)


def test_detection_metrics_bundle_and_confusion():
    dets = [
        _det([[10, 10, 50, 50], [0, 0, 5, 5]], [0.9, 0.3], [0, 1], [[10, 10, 50, 50]], [0]),
    ]
    res = detection_metrics(dets, [0, 1])
    # one TP (class 0) and one FP (class 1 has no ground truth)
    assert res["precision@0.25"] == pytest.approx(0.5)
    assert res["recall@0.25"] == pytest.approx(1.0)
    cm = confusion_matrix(dets, [0, 1], conf_threshold=0.25)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # gt class 0 matched by pred class 0
    assert cm[2, 1] == 1  # background gt, predicted class 1 -> false positive
