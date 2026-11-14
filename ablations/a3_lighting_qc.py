"""Ablation A3: Image QC Gate Robustness under Lighting Drift.

Hypothesis:
Lighting shifts (overexposure, underexposure, directional shifts) cause raw
anomaly detectors to cross the decision boundary and trigger false alarms.
The deterministic QC Gate converts corrupted frames to REVIEW ('re-image')
rather than false FAILs, maintaining station reliability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from veriqc.contract.qc import QualityCalibration, assess_quality
from veriqc.contract.triage import TriageConfig, triage
from veriqc.vision.gate import AnomalyGate

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "tables"


def evaluate_lighting_robustness(
    gate: AnomalyGate,
    quality_calib: QualityCalibration,
    test_scenes: dict[str, list[Image.Image]],  # condition -> list of good/normal images
    tau_fail: float = 0.50,
    tau_review: float = 0.30,
) -> dict:
    """Evaluate spurious alarms on normal parts under lighting shifts with QC gate ON vs OFF."""
    results = {}
    thresholds = {"review": tau_review, "fail": tau_fail}
    triage_cfg = TriageConfig(rule="score_only")

    for condition, images in test_scenes.items():
        gate_res = gate.score(images)
        scores = gate_res.image_scores

        # QC Gate OFF: direct thresholding
        raw_fails = int(np.sum(scores > tau_fail))
        raw_reviews = int(np.sum((scores > tau_review) & (scores <= tau_fail)))
        raw_passes = int(np.sum(scores <= tau_review))

        # QC Gate ON: assess quality first
        qc_fails = 0
        qc_reviews = 0
        qc_passes = 0
        intercepted_by_qc = 0

        for i, img in enumerate(images):
            s = float(scores[i])
            q_rep = assess_quality(img, quality_calib)
            if not q_rep.ok:
                intercepted_by_qc += 1
                qc_reviews += 1
                continue

            dec = triage(anomaly_score=s, detection=None, thresholds=thresholds, config=triage_cfg, qc_ok=True)
            if dec.verdict == "FAIL":
                qc_fails += 1
            elif dec.verdict == "REVIEW":
                qc_reviews += 1
            else:
                qc_passes += 1

        n_total = len(images)
        results[condition] = {
            "n_images": n_total,
            "gate_off": {
                "false_fail_count": raw_fails,
                "false_fail_rate": round(raw_fails / n_total, 4) if n_total else 0.0,
            },
            "gate_on": {
                "false_fail_count": qc_fails,
                "false_fail_rate": round(qc_fails / n_total, 4) if n_total else 0.0,
                "intercepted_to_review": intercepted_by_qc,
            },
            "spurious_fail_reduction": raw_fails - qc_fails,
        }

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS / "a3_lighting_qc.json"))
    args = ap.parse_args()

    print("Running Ablation A3: QC Gate Lighting Shift Robustness...")
    print("Ablation A3 implementation ready.")


if __name__ == "__main__":
    main()

