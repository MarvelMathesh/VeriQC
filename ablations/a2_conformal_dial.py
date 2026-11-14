"""Ablation A2: Decision Boundary -- Conformal Risk Control vs. Fixed Percentile.

Hypothesis:
A fixed score cutoff (e.g. 97.5th percentile) cannot guarantee a false-alarm
bound on unseen parts and drifts across categories.
Split-conformal calibration provides a statistically rigorous error budget dial
with Clopper-Pearson 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from veriqc.metrics.conformal import (
    conformal_threshold,
    cp_upper_fpr,
    min_n_for_target,
)

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "tables"


def evaluate_conformal_dial(
    scores_calib: np.ndarray,
    scores_test_normal: np.ndarray,
    scores_test_defect: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict:
    """Compare conformal calibration vs fixed 97.5th percentile threshold."""
    results = {"conformal": [], "fixed_percentile": {}}

    # 1. Fixed 97.5th percentile baseline
    tau_fixed = float(np.percentile(scores_calib, 97.5))
    empirical_fpr_fixed = float(np.mean(scores_test_normal > tau_fixed))
    miss_rate_fixed = float(np.mean(scores_test_defect <= tau_fixed))

    results["fixed_percentile"] = {
        "threshold": round(tau_fixed, 4),
        "empirical_fpr": round(empirical_fpr_fixed, 4),
        "miss_rate": round(miss_rate_fixed, 4),
    }

    # 2. Conformal dial across target alphas
    n_calib = len(scores_calib)
    for alpha in alphas:
        res_conf = conformal_threshold(scores_calib, alpha)
        if not res_conf.achievable:
            results["conformal"].append(
                {
                    "target_alpha": alpha,
                    "achievable": False,
                    "min_n_required": min_n_for_target(alpha),
                    "n_calib_available": n_calib,
                }
            )
            continue

        tau_conf = res_conf.threshold
        emp_fpr = float(np.mean(scores_test_normal > tau_conf))
        miss_rate = float(np.mean(scores_test_defect <= tau_conf))
        k_alarms = int(np.sum(scores_test_normal > tau_conf))
        cp_upper = cp_upper_fpr(k_alarms, len(scores_test_normal), confidence=0.95)

        results["conformal"].append(
            {
                "target_alpha": alpha,
                "achievable": True,
                "threshold": round(float(tau_conf), 4),
                "empirical_fpr": round(emp_fpr, 4),
                "cp_95_upper_fpr": round(float(cp_upper), 4) if cp_upper else None,
                "miss_rate": round(miss_rate, 4),
                "guarantee_holds": emp_fpr <= alpha,
            }
        )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS / "a2_conformal_dial.json"))
    args = ap.parse_args()

    print("Running Ablation A2: Conformal Triage Dial vs Fixed Percentile...")
    # Synthetic smoke demonstration
    rng = np.random.default_rng(42)
    s_calib = rng.normal(loc=0.2, scale=0.05, size=350)
    s_test_norm = rng.normal(loc=0.2, scale=0.05, size=200)
    s_test_def = rng.normal(loc=0.6, scale=0.1, size=100)

    res = evaluate_conformal_dial(s_calib, s_test_norm, s_test_def)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

