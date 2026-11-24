#!/usr/bin/env python3
"""Calibrate one or all profiles on a held-out normal population."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.calibration import ChannelCalibration, ProfileCalibration, fit_region_calibration  # noqa: E402
from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import (  # noqa: E402
    dataset_of,
    load_manifest,
    neu_boxes,
    split_by_part,
)
from veriqc.models.torch_detector import Detector  # noqa: E402
from veriqc.models.torch_patchcore import PatchCore  # noqa: E402
from veriqc.models.torch_segmenter import Segmenter  # noqa: E402


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _steel_class_thresholds(profile, detector, items: list[dict]) -> dict[str, float]:
    """Calibrate per-class report thresholds on the development boxes."""

    annotation_root = REPO / "data" / "neu" / "NEU-DET" / "train" / "annotations"
    records: dict[str, list[tuple[float, bool]]] = {name: [] for name in profile.known.classes}
    gt_counts: dict[str, int] = {name: 0 for name in profile.known.classes}
    for item in items:
        xml = annotation_root / f"{Path(item['image']).stem}.xml"
        if not xml.exists():
            continue
        gt = []
        for class_id, x1, y1, x2, y2 in neu_boxes(xml):
            if 0 <= class_id < len(profile.known.classes):
                name = profile.known.classes[class_id]
                gt.append((name, (x1, y1, x2, y2)))
                gt_counts[name] += 1
        with Image.open(item["image"]) as image:
            detections, _ = detector.predict(image.convert("RGB"))
        for detection in detections:
            best = max(
                (
                    _iou(detection.bbox, box)
                    for name, box in gt
                    if name == detection.class_name
                ),
                default=0.0,
            )
            records.setdefault(detection.class_name, []).append(
                (float(detection.confidence), best >= 0.5)
            )

    thresholds: dict[str, float] = {}
    for name in profile.known.classes:
        rows = records.get(name, [])
        gt_count = gt_counts.get(name, 0)
        if not rows or gt_count == 0:
            thresholds[name] = float(profile.known.report_threshold)
            continue
        candidates = sorted({confidence for confidence, _ in rows}, reverse=True)
        best = (0.0, float(profile.known.report_threshold))
        for threshold in candidates:
            selected = [(confidence, correct) for confidence, correct in rows if confidence >= threshold]
            tp = sum(1 for _, correct in selected if correct)
            fp = len(selected) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(gt_count, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            if precision >= 0.85 and f1 > best[0]:
                best = (f1, float(threshold))
        if best[0] == 0.0:
            thresholds[name] = float(profile.known.report_threshold)
        else:
            thresholds[name] = best[1]
    return thresholds


def calibrate_detector(profile_path: Path, device: str) -> ProfileCalibration:
    profile = load_profile(profile_path, validate=False)
    manifest = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")
    detector = Detector.from_weights(
        profile.known.path,
        profile.known.classes,
        input_size=int(profile.known.input_size or 640),
        conf_threshold=profile.known.conf_threshold,
        device=device,
    )
    scores = []
    for item in manifest["items"]["val"]:
        with Image.open(item["image"]) as image:
            detections, _ = detector.predict(image.convert("RGB"))
        scores.append(max((d.confidence for d in detections), default=0.0))
    values = np.asarray(scores, dtype=np.float64)
    fail = float(np.quantile(values, 0.10))
    # A raw-score review band must sit below the FAIL threshold: scores in
    # [review, fail) are sent to a human, scores below review pass.  Choose the
    # highest such threshold that preserves the screening recall target while
    # keeping the review workload bounded.
    candidates = np.unique(values[values < fail])
    review = float(np.min(values))
    for candidate in np.sort(candidates)[::-1]:
        non_pass = float((values >= candidate).mean())
        review_rate = float(((values >= candidate) & (values < fail)).mean())
        if non_pass >= 0.95 and review_rate <= 0.10:
            review = float(candidate)
            break
    class_thresholds = _steel_class_thresholds(profile, detector, manifest["items"]["val"])
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="known_only",
        alpha_fail=0.10,
        alpha_review=0.30,
        class_thresholds=class_thresholds,
        screening_fail=fail,
        screening_review=review,
        sample_sizes={"known_cal": len(scores)},
        note=(
            "Known-class screening on NEU-DET. FAIL is the 10th percentile of the "
            "development scores; REVIEW is the highest lower threshold that keeps "
            "non-PASS recall at or above 95% with a review rate at or below 10%. "
            "Per-class report thresholds are calibrated on matched development boxes. The "
            "10% development split is also used for early stopping, so the thresholds "
            "are development estimates; the source has no normal steel population, so "
            "no false-alarm claim is made."
        ),
    )


def calibrate_segmenter(profile_path: Path, device: str) -> ProfileCalibration:
    profile = load_profile(profile_path, validate=False)
    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    fold = manifest["folds"][0]
    normals = [item for item in fold["train"] if not item["is_defect"]]
    _fit, cal = split_by_part(normals, 20261121)
    segmenter = Segmenter.from_checkpoint(profile.known.path, device=device)
    known_scores = []
    region_maps = []
    for item in cal:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            prob, _ = segmenter.predict_proba(rgb)
            known_scores.append(float(prob.max()))
            region_maps.append(prob)
    known_channel = ChannelCalibration(scores=known_scores)
    region = fit_region_calibration(
        region_maps,
        min_area_px=profile.known.min_area_px,
        prior_threshold=float(profile.known.score_threshold or 0.5),
        target_false_regions=0.5,
    )
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="known_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        known=known_channel,
        region=region,
        screening_fail=known_channel.threshold_for_alpha(0.05),
        screening_review=known_channel.threshold_for_alpha(0.10),
        sample_sizes={"known_cal": len(known_scores), "region_cal": len(region.scores)},
        note=(
            "Kolektor fold 0, portrait 384x1024 segmenter. The 15% physical-part holdout "
            "is used for early stopping and calibration; this shared split is a disclosed "
            "selection limitation. The anomaly channel added no recall at the 5% operating "
            "point and increased false alarms. Localisation uses a normal-region calibrated "
            "threshold rather than the image-level screening threshold."
        ),
    )


def calibrate_anomaly(profile_path: Path, device: str) -> ProfileCalibration:
    profile = load_profile(profile_path, validate=False)
    dataset, category = dataset_of(profile)
    if dataset != "mvtec" or category is None:
        raise ValueError(f"profile {profile.profile_id!r} is not a MVTec anomaly profile")
    if profile.anomaly is None:
        raise ValueError(f"profile {profile.profile_id!r} has no anomaly checkpoint")
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    items = [item for item in manifest["categories"][category]["items"] if item["split"] == "cal"]
    anomaly = PatchCore.load(profile.anomaly.path, device=device)
    scores = []
    region_maps = []
    for item in items:
        with Image.open(item["image"]) as image:
            result = anomaly.score(image.convert("RGB"))
            scores.append(float(result.image_score))
            region_maps.append(result.heatmap)
    image_channel = ChannelCalibration(scores=scores)
    region = fit_region_calibration(
        region_maps,
        prior_threshold=image_channel.threshold_for_alpha(0.10),
        target_false_regions=0.5,
    )
    localisation = (
        "localisation uses a normal-region calibrated threshold."
        if region.scores
        else (
            "no normal-region exceedance was observed at the image-level prior; "
            "the threshold stays at that prior and region p-values are 1.0."
        )
    )
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="anomaly_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        anomaly=image_channel,
        region=region,
        sample_sizes={"anomaly_cal": len(scores), "region_cal": len(region.scores)},
        note=(
            f"MVTec AD {category}: image calibration on a disjoint 20% of train/good; "
            f"{localisation}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="all")
    import torch

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    profiles = sorted((REPO / "configs" / "profiles").glob("*.yaml"))
    if args.profile != "all":
        profiles = [REPO / "configs" / "profiles" / f"{args.profile}.yaml"]
    for path in profiles:
        profile = load_profile(path, validate=False)
        if profile.task == "detect":
            calibration = calibrate_detector(path, args.device)
        elif profile.task == "segment":
            calibration = calibrate_segmenter(path, args.device)
        elif profile.task == "anomaly":
            calibration = calibrate_anomaly(path, args.device)
        else:
            continue
        calibration.save(profile.calibration_path)
        print(profile.profile_id, "->", profile.calibration_path, calibration.sample_sizes)


if __name__ == "__main__":
    main()
