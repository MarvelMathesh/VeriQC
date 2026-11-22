#!/usr/bin/env python3
"""Compare a fixed score threshold with the calibrated operating point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402


def scores(pipeline: Pipeline, items: list[dict], channel: str) -> tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for item in items:
        with Image.open(item["image"]) as image:
            result = pipeline.analyze(image.convert("RGB"), "CALIBRATION").result
        values.append(float(getattr(result, f"{channel}_score") or 0.0))
        labels.append(item["is_defect"])
    return np.asarray(values), np.asarray(labels)


def rates(values: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    flags = values >= threshold
    return {
        "threshold": round(float(threshold), 6),
        "recall": round(float(flags[labels].mean()), 4) if labels.any() else None,
        "false_alarm_rate": round(float(flags[~labels].mean()), 4) if (~labels).any() else None,
    }


def main() -> None:
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = []
    for profile_id, channel, items in (
        ("textile_carpet", "anomaly", None),
        ("commutator", "known", None),
    ):
        profile = load_profile(REPO / "configs" / "profiles" / f"{profile_id}.yaml")
        pipeline = Pipeline(profile, device=args.device)
        if profile_id == "textile_carpet":
            manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
            items = [item for item in manifest["categories"]["carpet"]["items"] if item["split"] == "test"]
            calibrated = pipeline.calibration.anomaly.threshold_for_alpha(pipeline.calibration.alpha_fail)
        else:
            manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
            items = manifest["folds"][0]["test"]
            calibrated = pipeline.calibration.screening_fail
        values, labels = scores(pipeline, items, channel)
        rows.append(
            {
                "profile_id": profile_id,
                "channel": channel,
                "fixed_0.5": rates(values, labels, 0.5),
                "calibrated": rates(values, labels, calibrated),
            }
        )
        print(rows[-1], flush=True)
    output = REPO / "reports" / "tables" / "ablation_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"arms": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
