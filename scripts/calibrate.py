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

from veriqc.calibration import ChannelCalibration, ProfileCalibration  # noqa: E402
from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import load_manifest, split_by_part  # noqa: E402
from veriqc.models import Detector, PatchCore, Segmenter  # noqa: E402

MVTEC_PROFILES = {
    "textile_carpet": "carpet",
    "packaging_bottle": "bottle",
    "pharma_capsule": "capsule",
    "automotive_metal_nut": "metal_nut",
    "electronics_transistor": "transistor",
}


def steel(profile_path: Path, device: str) -> ProfileCalibration:
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
    ordered = sorted(scores)
    fail = float(np.quantile(ordered, 0.10))
    review = float(np.quantile(ordered, 0.30))
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="known_only",
        alpha_fail=0.10,
        alpha_review=0.30,
        screening_fail=fail,
        screening_review=review,
        sample_sizes={"known_cal": len(scores)},
        note=(
            "Known-class screening on NEU-DET. The 10% development split is also used "
            "for early stopping, so this threshold is a development estimate; the shared "
            "split is a disclosed limitation. The source has no normal steel population, "
            "so no false-alarm claim is made."
        ),
    )


def kolektor(profile_path: Path, device: str) -> ProfileCalibration:
    profile = load_profile(profile_path, validate=False)
    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    fold = manifest["folds"][0]
    normals = [item for item in fold["train"] if not item["is_defect"]]
    _fit, cal = split_by_part(normals, 20261121)
    segmenter = Segmenter.from_checkpoint(profile.known.path, device=device)
    known_scores = []
    for item in cal:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            prob, _ = segmenter.predict_proba(rgb)
            known_scores.append(float(prob.max()))
    known_channel = ChannelCalibration(scores=known_scores)
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="known_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        known=known_channel,
        screening_fail=known_channel.threshold_for_alpha(0.05),
        screening_review=known_channel.threshold_for_alpha(0.10),
        sample_sizes={"known_cal": len(known_scores)},
        note=(
            "Kolektor fold 0, portrait 384x1024 segmenter. The 15% physical-part holdout "
            "is used for early stopping and calibration; this shared split is a disclosed "
            "selection limitation. The anomaly channel added no recall at the 5% operating "
            "point and increased false alarms."
        ),
    )


def anomaly_profile(profile_path: Path, device: str) -> ProfileCalibration:
    profile = load_profile(profile_path, validate=False)
    category = MVTEC_PROFILES[profile.profile_id]
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    items = [item for item in manifest["categories"][category]["items"] if item["split"] == "cal"]
    anomaly = PatchCore.load(profile.anomaly.path, device=device)
    scores = []
    for item in items:
        with Image.open(item["image"]) as image:
            scores.append(float(anomaly.score(image.convert("RGB")).image_score))
    return ProfileCalibration(
        profile_id=profile.profile_id,
        rule="anomaly_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        anomaly=ChannelCalibration(scores=scores),
        sample_sizes={"anomaly_cal": len(scores)},
        note=f"MVTec AD {category}: calibration on a disjoint 20% of train/good.",
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
        if profile.profile_id == "steel_surface":
            calibration = steel(path, args.device)
        elif profile.profile_id == "commutator":
            calibration = kolektor(path, args.device)
        elif profile.profile_id in MVTEC_PROFILES:
            calibration = anomaly_profile(path, args.device)
        else:
            continue
        calibration.save(profile.calibration_path)
        print(profile.profile_id, "->", profile.calibration_path, calibration.sample_sizes)


if __name__ == "__main__":
    main()
