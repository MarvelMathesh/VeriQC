#!/usr/bin/env python3
"""Reproduce the Kolektor channel ablation from the saved models.

The production commutator profile is segmentation-only.  This experiment keeps
the anomaly channel to show why: at the 5% operating point it added no recall
and increased false alarms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.calibration import ChannelCalibration  # noqa: E402
from veriqc.datasets import load_manifest, split_by_part  # noqa: E402
from veriqc.models.torch_patchcore import PatchCore  # noqa: E402
from veriqc.models.torch_segmenter import Segmenter  # noqa: E402


def scores(segmenter: Segmenter, anomaly: PatchCore, items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    known = []
    unknown = []
    for item in items:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            prob, _ = segmenter.predict_proba(rgb)
            known.append(float(prob.max()))
            unknown.append(float(anomaly.score(rgb).image_score))
    return np.asarray(known), np.asarray(unknown)


def rates(flags: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "recall": round(float(flags[labels].mean()), 4) if labels.any() else None,
        "false_alarm_rate": round(float(flags[~labels].mean()), 4) if (~labels).any() else None,
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    fold = manifest["folds"][0]
    normals = [item for item in fold["train"] if not item["is_defect"]]
    _fit, cal = split_by_part(normals, 20261121)
    segmenter = Segmenter.from_checkpoint(
        REPO / "models" / "profiles" / "commutator" / "fold0_unet.pt", device=args.device
    )
    anomaly = PatchCore.load(
        REPO / "models" / "profiles" / "commutator" / "patchcore.pt", device=args.device
    )
    cal_known, cal_anomaly = scores(segmenter, anomaly, cal)
    test_known, test_anomaly = scores(segmenter, anomaly, fold["test"])
    labels = np.asarray([item["is_defect"] for item in fold["test"]])
    known_channel = ChannelCalibration(scores=cal_known.tolist())
    anomaly_channel = ChannelCalibration(scores=cal_anomaly.tolist())
    fused_channel = ChannelCalibration(
        scores=[1.0 - min(known_channel.p_value(k), anomaly_channel.p_value(a)) for k, a in zip(cal_known, cal_anomaly)]
    )
    known_flags = test_known >= known_channel.threshold_for_alpha(0.05)
    anomaly_flags = test_anomaly >= anomaly_channel.threshold_for_alpha(0.05)
    fused_flags = np.asarray(
        [fused_channel.p_value(1.0 - min(known_channel.p_value(k), anomaly_channel.p_value(a))) <= 0.05 for k, a in zip(test_known, test_anomaly)]
    )
    payload = {
        "profile_id": "commutator",
        "fold": 0,
        "alpha_fail": 0.05,
        "arms": [
            {"arm": "segmentation-only", "threshold": known_channel.threshold_for_alpha(0.05), **rates(known_flags, labels)},
            {"arm": "anomaly-only", "threshold": anomaly_channel.threshold_for_alpha(0.05), **rates(anomaly_flags, labels)},
            {"arm": "dual", **rates(fused_flags, labels)},
        ],
    }
    output = REPO / "reports" / "tables" / "ablation_commutator_channels.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
