"""Detection and classification metrics.

Everything here is implemented in-house on purpose: the anomaly gate, the
supervised detector and the triage engine are all compared against the *same*
metric code, so a candidate never wins because it was scored by a different
(usually more favourable) evaluator.

Conventions
-----------
* Boxes are ``(x1, y1, x2, y2)`` in absolute pixel coordinates, ``x2 > x1``.
* Labels are integer class ids, zero based.
* Scores are confidences in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Detection",
    "average_precision",
    "mean_average_precision",
    "detection_metrics",
    "confusion_matrix",
]


@dataclass
class Detection:
    """One image's worth of predictions and ground truth."""

    pred_boxes: np.ndarray  # (N, 4) float, xyxy
    pred_scores: np.ndarray  # (N,) float
    pred_labels: np.ndarray  # (N,) int
    gt_boxes: np.ndarray  # (M, 4) float, xyxy
    gt_labels: np.ndarray  # (M,) int

    def __post_init__(self) -> None:
        self.pred_boxes = np.asarray(self.pred_boxes, dtype=np.float64).reshape(-1, 4)
        self.pred_scores = np.asarray(self.pred_scores, dtype=np.float64).reshape(-1)
        self.pred_labels = np.asarray(self.pred_labels, dtype=np.int64).reshape(-1)
        self.gt_boxes = np.asarray(self.gt_boxes, dtype=np.float64).reshape(-1, 4)
        self.gt_labels = np.asarray(self.gt_labels, dtype=np.int64).reshape(-1)
        if not (len(self.pred_boxes) == len(self.pred_scores) == len(self.pred_labels)):
            raise ValueError("prediction arrays disagree in length")
        if len(self.gt_boxes) != len(self.gt_labels):
            raise ValueError("ground-truth arrays disagree in length")


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    lt = np.maximum(a[..., :2], b[..., :2])
    rb = np.minimum(a[..., 2:], b[..., 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(
    tp: np.ndarray,
    fp: np.ndarray,
    n_gt: int,
    style: str = "voc",
    n_points: int = 101,
) -> float:
    """Average precision from ranked TP/FP counts.

    ``style='voc'``   all-point interpolation (the convention behind the
                       published NEU-DET mAP50 numbers, so our detector table is
                       comparable with them).
    ``style='coco'``  101-point interpolated AP on an 11x13 IoU grid is done
                       by the caller; here it is the per-IoU AP with 101-point
                       recall interpolation.
    """
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp.astype(np.float64))
    fp_cum = np.cumsum(fp.astype(np.float64))
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    if style == "voc":
        # All-point interpolation: make precision monotonically decreasing by
        # taking the max to the right, then integrate over recall.
        mpre = np.concatenate(([0.0], precision, [0.0]))
        mrec = np.concatenate(([0.0], recall, [1.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    if style == "coco":
        # 101-point interpolated AP.
        rec_thr = np.linspace(0.0, 1.0, n_points)
        prec = np.interp(rec_thr, recall, precision, left=precision[0], right=0.0)
        # enforce monotone envelope
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        return float(np.mean(prec))

    raise ValueError(f"unknown style {style!r}")


def _match_class(
    det: Detection,
    cls: int,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy score-ordered matching of predictions of one class to gt of same class.

    Returns ``(tp, fp, order)`` where ``order`` indexes the class's
    predictions sorted by descending score; ``tp``/``fp`` are aligned with
    ``order``.  Callers must use ``order`` to align the scores too.
    """
    p_mask = det.pred_labels == cls
    p_boxes = det.pred_boxes[p_mask]
    p_scores = det.pred_scores[p_mask]
    g_mask = det.gt_labels == cls
    g_boxes = det.gt_boxes[g_mask]

    order = np.argsort(-p_scores, kind="stable")
    p_boxes = p_boxes[order]
    tp = np.zeros(len(p_boxes), dtype=bool)
    fp = np.zeros(len(p_boxes), dtype=bool)
    if len(g_boxes) == 0:
        fp[:] = True
        return tp, fp, order
    if len(p_boxes) == 0:
        return tp, fp, order

    ious = _iou_matrix(p_boxes, g_boxes)
    used = np.zeros(len(g_boxes), dtype=bool)
    for i in range(len(p_boxes)):
        best_j, best_iou = -1, 0.0
        for j in range(len(g_boxes)):
            if used[j]:
                continue
            if ious[i, j] > best_iou:
                best_iou, best_j = ious[i, j], j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = True
            used[best_j] = True
        else:
            fp[i] = True
    return tp, fp, order


def mean_average_precision(
    detections: list[Detection],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    style: str = "voc",
    area_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """mAP over classes.

    ``area_ranges`` optionally restricts *ground truth* to an area band and
    ignores predictions that do not overlap a gt inside that band (COCO style
    area filtering), used for the mAP50-95 small/medium/large breakdown.
    """
    aps: list[float] = []
    per_class: dict[int, float] = {}
    for cls in class_ids:
        n_gt = 0
        # (score, tp, fp) for every prediction of this class, across all images
        scored: list[tuple[float, bool, bool]] = []
        for det in detections:
            if area_ranges is None:
                d = det
            else:
                lo, hi = area_ranges["gt"]
                keep = (det.gt_labels == cls) & (
                    (det.gt_boxes[:, 2] - det.gt_boxes[:, 0])
                    * (det.gt_boxes[:, 3] - det.gt_boxes[:, 1])
                    >= lo
                ) & (
                    (det.gt_boxes[:, 2] - det.gt_boxes[:, 0])
                    * (det.gt_boxes[:, 3] - det.gt_boxes[:, 1])
                    < hi
                )
                g_boxes = det.gt_boxes[keep]
                g_labels = det.gt_labels[keep]
                # predictions: keep only those overlapping a kept gt (or all, if
                # the image has no gt of this class -> they are all FP)
                p_mask = det.pred_labels == cls
                p_boxes = det.pred_boxes[p_mask]
                p_scores = det.pred_scores[p_mask]
                if len(g_boxes):
                    ious = _iou_matrix(p_boxes, g_boxes)
                    overlap = ious.max(axis=1) if len(p_boxes) else np.zeros(0)
                    keep_p = overlap > 0
                else:
                    keep_p = np.ones(len(p_boxes), dtype=bool)
                d = Detection(
                    pred_boxes=p_boxes[keep_p],
                    pred_scores=p_scores[keep_p],
                    pred_labels=cls * np.ones(int(keep_p.sum()), dtype=np.int64),
                    gt_boxes=g_boxes,
                    gt_labels=g_labels,
                )
            tp, fp, order = _match_class(d, cls, iou_threshold)
            p_scores = d.pred_scores[d.pred_labels == cls][order]
            for s, t, f in zip(p_scores, tp, fp):
                scored.append((float(s), bool(t), bool(f)))
            n_gt += int((d.gt_labels == cls).sum())
        if n_gt == 0:
            continue
        if not scored:
            per_class[cls] = 0.0
            aps.append(0.0)
            continue
        scored.sort(key=lambda x: -x[0])
        tp_all = np.array([t for _, t, _ in scored])
        fp_all = np.array([f for _, _, f in scored])
        ap = average_precision(tp_all, fp_all, n_gt, style=style)
        per_class[cls] = ap
        aps.append(ap)
    out = {"mAP": float(np.mean(aps)) if aps else float("nan")}
    for cls, ap in per_class.items():
        out[f"AP_class_{cls}"] = ap
    return out


def detection_metrics(
    detections: list[Detection],
    class_ids: list[int],
    iou_thresholds: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10)),
    style: str = "voc",
) -> dict[str, float]:
    """Full detector metric bundle: mAP50, mAP50-95, precision/recall at a fixed conf."""
    res: dict[str, float] = {}
    res.update(mean_average_precision(detections, class_ids, 0.5, style=style))
    aps = []
    for thr in iou_thresholds:
        r = mean_average_precision(detections, class_ids, thr, style="coco")
        aps.append(r["mAP"])
    res["mAP50_95"] = float(np.mean(aps)) if aps else float("nan")

    # precision / recall / F1 at a fixed operating confidence (0.25, the
    # conventional detection operating point).
    tp_tot = fp_tot = fn_tot = 0
    for det in detections:
        for cls in class_ids:
            tp, fp, order = _match_class(det, cls, 0.5)
            scores = det.pred_scores[det.pred_labels == cls][order]
            keep = scores >= 0.25
            tp_tot += int(tp[keep].sum())
            fp_tot += int(fp[keep].sum())
            fn_tot += int(((det.gt_labels == cls).sum()) - tp[keep].sum())
    prec = tp_tot / max(tp_tot + fp_tot, 1)
    rec = tp_tot / max(tp_tot + fn_tot, 1)
    res["precision@0.25"] = prec
    res["recall@0.25"] = rec
    res["f1@0.25"] = 2 * prec * rec / max(prec + rec, 1e-12)
    res["tp@0.25"] = float(tp_tot)
    res["fp@0.25"] = float(fp_tot)
    res["fn@0.25"] = float(fn_tot)
    return res


def confusion_matrix(
    detections: list[Detection],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """(C+1)x(C+1) confusion matrix; last row/col are background (FP/FN).

    Row = ground truth, column = prediction, both in class-id order with a
    trailing 'no object' bucket.
    """
    k = len(class_ids)
    idx = {c: i for i, c in enumerate(class_ids)}
    cm = np.zeros((k + 1, k + 1), dtype=np.int64)
    for det in detections:
        # match greedily across all classes by score
        order = np.argsort(-det.pred_scores)
        used = np.zeros(len(det.gt_boxes), dtype=bool)
        assignments: list[tuple[int, int]] = []  # (pred_idx or -1, gt_idx or -1)
        for i in order:
            if det.pred_scores[i] < conf_threshold:
                continue
            if len(det.gt_boxes) == 0:
                assignments.append((i, -1))
                continue
            ious = _iou_matrix(det.pred_boxes[i : i + 1], det.gt_boxes)[0]
            ious[used] = -1.0
            j = int(np.argmax(ious))
            if ious[j] >= iou_threshold:
                used[j] = True
                assignments.append((i, j))
            else:
                assignments.append((i, -1))
        for i, j in assignments:
            pred_c = idx.get(int(det.pred_labels[i]), k)
            gt_c = idx.get(int(det.gt_labels[j]), k) if j >= 0 else k
            cm[gt_c, pred_c] += 1
        # unmatched gt
        for j in range(len(det.gt_boxes)):
            if not used[j]:
                cm[idx.get(int(det.gt_labels[j]), k), k] += 1
    return cm
