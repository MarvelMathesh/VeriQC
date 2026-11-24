#!/usr/bin/env python3
"""Evaluate one profile with task-appropriate metrics and calibrated verdicts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import test_items  # noqa: E402
from veriqc.metrics import au_pro, average_precision, image_auroc, pixel_auroc  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402


def _rate(count: int, total: int) -> float | None:
    return round(float(count) / total, 4) if total else None


def evaluate_items(
    pipeline: Pipeline,
    items: list[dict],
    *,
    need_masks: bool = False,
    metric_size: tuple[int, int] = (512, 512),
    collect_scores: bool = False,
) -> dict:
    verdicts: list[str] = []
    labels: list[bool] = []
    known_scores: list[float] = []
    anomaly_scores: list[float] = []
    anomaly_maps: list[np.ndarray] = []
    known_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    latencies: list[float] = []
    per_image: list[dict] = []
    qc_review = 0
    want_anomaly_maps = need_masks and pipeline.profile.anomaly is not None
    want_known_maps = need_masks and pipeline.segmenter is not None

    for item in items:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            analysis = pipeline.analyze(rgb, str(item.get("part") or "EVAL"))
            size = rgb.size
        result = analysis.result
        verdicts.append(result.verdict)
        labels.append(bool(item["is_defect"]))
        known_scores.append(float(result.known_score) if result.known_score is not None else 0.0)
        anomaly_scores.append(
            float(result.anomaly_score) if result.anomaly_score is not None else 0.0
        )
        latencies.append(
            result.latency_ms.get("known_ms", 0.0)
            + result.latency_ms.get("anomaly_ms", 0.0)
        )
        if not result.acquisition.get("qc_ok", True):
            qc_review += 1
        if collect_scores:
            per_image.append(
                {
                    "part_id": str(item.get("part") or "EVAL"),
                    "defect_class": item.get("defect_class"),
                    "label": bool(item["is_defect"]),
                    "verdict": result.verdict,
                    "qc_ok": bool(result.acquisition.get("qc_ok", True)),
                    "known_score": result.known_score,
                    "anomaly_score": result.anomaly_score,
                }
            )

        if need_masks:
            if item.get("label"):
                with Image.open(item["label"]) as mask:
                    native_mask = np.asarray(mask.convert("L")) > 127
            else:
                native_mask = np.zeros((size[1], size[0]), dtype=bool)
            masks.append(
                cv2.resize(
                    native_mask.astype(np.uint8), metric_size, interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
            if want_anomaly_maps:
                heatmap = analysis.heatmap
                source = (
                    heatmap.astype(np.float32)
                    if heatmap is not None
                    else np.zeros((size[1], size[0]), dtype=np.float32)
                )
                anomaly_maps.append(
                    cv2.resize(source, metric_size, interpolation=cv2.INTER_LINEAR)
                )
            if want_known_maps:
                probability = analysis.known_mask
                source = (
                    probability.astype(np.float32)
                    if probability is not None
                    else np.zeros((size[1], size[0]), dtype=np.float32)
                )
                known_maps.append(
                    cv2.resize(source, metric_size, interpolation=cv2.INTER_LINEAR)
                )

    verdicts_array = np.asarray(verdicts)
    labels_array = np.asarray(labels, dtype=bool)
    known_array = np.asarray(known_scores)
    anomaly_array = np.asarray(anomaly_scores)
    non_pass = verdicts_array != "PASS"
    fail = verdicts_array == "FAIL"

    defect_count = int(labels_array.sum())
    normal_count = int((~labels_array).sum())
    detected_non_pass = int((non_pass & labels_array).sum())
    detected_fail = int((fail & labels_array).sum())
    false_alarms_non_pass = int((non_pass & ~labels_array).sum())
    false_alarms_fail = int((fail & ~labels_array).sum())

    result = {
        "images": len(items),
        "defect_images": defect_count,
        "normal_images": normal_count,
        "verdict_counts": {
            "PASS": int((verdicts_array == "PASS").sum()),
            "REVIEW": int((verdicts_array == "REVIEW").sum()),
            "FAIL": int((verdicts_array == "FAIL").sum()),
        },
        "confusion": {
            "non_pass": {
                "detected": detected_non_pass,
                "missed": defect_count - detected_non_pass,
                "false_alarms": false_alarms_non_pass,
                "true_negatives": normal_count - false_alarms_non_pass,
            },
            "fail": {
                "detected": detected_fail,
                "missed": defect_count - detected_fail,
                "false_alarms": false_alarms_fail,
                "true_negatives": normal_count - false_alarms_fail,
            },
        },
        "known_score_auroc": (
            round(image_auroc(known_array, labels_array), 4)
            if pipeline.profile.known is not None
            and labels_array.any()
            and (~labels_array).any()
            else None
        ),
        "anomaly_score_auroc": (
            round(image_auroc(anomaly_array, labels_array), 4)
            if pipeline.profile.anomaly is not None
            and labels_array.any()
            and (~labels_array).any()
            else None
        ),
        "detection_rate_non_pass": _rate(detected_non_pass, defect_count),
        "detection_rate_fail": _rate(detected_fail, defect_count),
        "false_alarm_rate_non_pass": _rate(false_alarms_non_pass, normal_count),
        "false_alarm_rate_fail": _rate(false_alarms_fail, normal_count),
        "review_rate": round(float((verdicts_array == "REVIEW").mean()), 4),
        "qc_review_count": qc_review,
        "qc_review_rate": round(qc_review / max(1, len(items)), 4),
        "latency_ms_median": round(float(np.median(latencies)), 2),
        "latency_ms_p95": round(float(np.quantile(latencies, 0.95)), 2),
    }
    if need_masks and masks:
        masks_array = np.stack(masks)
        if anomaly_maps:
            maps_array = np.stack(anomaly_maps)
            result.update(
                {
                    "anomaly_pixel_auroc": round(pixel_auroc(maps_array, masks_array), 4),
                    "anomaly_pixel_ap": round(average_precision(maps_array, masks_array), 4),
                    "anomaly_au_pro": round(au_pro(masks_array, maps_array), 4),
                }
            )
        if known_maps:
            maps_array = np.stack(known_maps)
            result.update(
                {
                    "known_pixel_auroc": round(pixel_auroc(maps_array, masks_array), 4),
                    "known_pixel_ap": round(average_precision(maps_array, masks_array), 4),
                    "known_au_pro": round(au_pro(masks_array, maps_array), 4),
                }
            )
    if collect_scores:
        result["per_image"] = per_image
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--dump-scores",
        action="store_true",
        help="write per-image scores for ROC/PR figures",
    )
    args = parser.parse_args()
    path = REPO / "configs" / "profiles" / f"{args.profile}.yaml"
    os.environ["VERIQC_THREADS"] = str(max(1, args.threads))
    profile = load_profile(path)
    pipeline = Pipeline(profile)
    items = test_items(profile)
    if not items:
        raise SystemExit(f"no test items for profile {args.profile}")
    need_masks = profile.task in ("segment", "anomaly")
    metric_size = (512, 1408) if profile.task == "segment" else (512, 512)
    result = evaluate_items(
        pipeline,
        items,
        need_masks=need_masks,
        metric_size=metric_size,
        collect_scores=args.dump_scores,
    )
    per_image = result.pop("per_image", [])
    result["profile_id"] = args.profile
    result["task"] = profile.task
    result["source"] = profile.source
    result["backend"] = pipeline.backend
    result["device"] = pipeline.device
    result["calibration"] = str(profile.calibration_path.relative_to(REPO))
    output = REPO / (args.out or f"reports/tables/evaluation_{args.profile}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.dump_scores:
        score_dir = REPO / "reports" / "tables" / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / f"{args.profile}.json"
        score_path.write_text(
            json.dumps(
                {
                    "schema": "veriqc.scores/1",
                    "profile_id": args.profile,
                    "task": profile.task,
                    "source": profile.source,
                    "score_kind": "known_score" if profile.task != "anomaly" else "anomaly_score",
                    "rows": per_image,
                },
                indent=2,
            )
            + "\n"
        )
        print(score_path.relative_to(REPO))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
