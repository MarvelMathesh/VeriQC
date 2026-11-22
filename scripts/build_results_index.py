#!/usr/bin/env python3
"""Assemble the machine-readable results index from the evidence tables.

Every headline number is read from a checked-in table rather than retyped.
Confidence intervals are exact Clopper-Pearson intervals on the underlying
counts.  Unmeasured items stay explicit so the index cannot be read as a
device claim.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.metrics import clopper_pearson  # noqa: E402

TABLES = REPO / "reports" / "tables"
PROFILES = (
    "steel_surface",
    "commutator",
    "textile_carpet",
    "packaging_bottle",
    "pharma_capsule",
    "automotive_metal_nut",
    "electronics_transistor",
)

PROTOCOL = {
    "steel_surface": (
        "NEU-DET official train/validation. The screening threshold is fit on a "
        "10% development split of official train (held out from gradient updates "
        "and also used for early stopping). The official validation split is the "
        "reported test set. No normal-steel false-alarm estimate exists."
    ),
    "commutator": (
        "KolektorSDD official JIM-2019 three-fold split by physical part; fold 0 "
        "is the operating point. A 15% physical-part holdout supplies both "
        "early-stopping and calibration scores; this shared split is a disclosed "
        "selection limitation. Portrait 384x1024 preprocessing is used."
    ),
}
for _profile in PROFILES[2:]:
    PROTOCOL[_profile] = (
        "MVTec AD official train/test; fit on 80% of train/good, calibrate on a "
        "disjoint 20%, report on the official test split."
    )


def load(name: str) -> dict:
    return json.loads((TABLES / name).read_text())


def interval(successes: int, n: int) -> list[float]:
    lower, upper = clopper_pearson(int(successes), int(n))
    return [round(lower, 4), round(upper, 4)]


def count_block(count: int, n: int) -> dict | None:
    if n <= 0:
        return None
    return {
        "count": int(count),
        "n": int(n),
        "rate": round(float(count) / n, 4),
        "ci95": interval(count, n),
    }


def _cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return names[0] if names else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def profile_record(profile_id: str) -> dict:
    config = yaml.safe_load((REPO / "configs" / "profiles" / f"{profile_id}.yaml").read_text())
    calibration = json.loads((REPO / "configs" / "calibration" / f"{profile_id}.json").read_text())
    evaluation = load(f"evaluation_{profile_id}.json")
    defect_n = int(evaluation["defect_images"])
    normal_n = int(evaluation["normal_images"])
    confusion = evaluation["confusion"]

    if profile_id == "steel_surface":
        detector = load("detector_neu.json")
        model_metrics = {
            "architecture": detector["architecture"],
            "input_size": detector["input_size"],
            "epochs": detector["epochs"],
            "seed": detector["seed"],
            "test": detector["test"],
            "weights_sha256": detector["weights_sha256"],
        }
    elif profile_id == "commutator":
        segmenter = load("segmenter_kolektor.json")
        model_metrics = {
            "architecture": "ResNet-18 U-Net",
            "input_size": [384, 1024],
            "metric_size": segmenter["metric_size"],
            "three_fold_mean": segmenter["mean"],
            "folds": [
                {
                    "fold": row["fold"],
                    "image_auroc": row["image_auroc"],
                    "pixel_auroc": row["pixel_auroc"],
                    "pixel_ap": row["pixel_ap"],
                    "au_pro": row["au_pro"],
                    "best_f1": row["best_f1"],
                }
                for row in segmenter["folds"]
            ],
            "fold_0_pixel": {
                "auroc": evaluation.get("known_pixel_auroc"),
                "ap": evaluation.get("known_pixel_ap"),
                "au_pro": evaluation.get("known_au_pro"),
            },
        }
    else:
        table = load("anomaly_mvtec.json")
        report = next(row for row in table["reports"] if row["profile_id"] == profile_id)
        model_metrics = {
            "architecture": "PatchCore, ResNet-18 layer2 image bank + layer3 localization bank",
            "input_size": 448,
            "bank_size": report["bank_size"],
            "image_auroc": report["test"]["image_auroc"],
            "pixel_auroc": report["test"]["pixel_auroc"],
            "pixel_ap": report["test"]["pixel_ap"],
            "au_pro": report["test"]["au_pro"],
            "weights_sha256": report["weights_sha256"],
        }

    latency = {
        "cuda_evaluation_ms": {
            "median": evaluation.get("latency_ms_median"),
            "p95": evaluation.get("latency_ms_p95"),
        }
    }
    cpu_path = TABLES / f"latency_{profile_id}.json"
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text())
        latency["cpu_station_ms"] = {
            "median": cpu["total_ms"]["median"],
            "p95": cpu["total_ms"]["p95"],
            "p99": cpu["total_ms"]["p99"],
            "peak_rss_mb": cpu["peak_rss_mb"],
            "device": cpu["device"],
            "threads": cpu["threads"],
        }

    evidence = [f"reports/tables/evaluation_{profile_id}.json"]
    if profile_id == "steel_surface":
        evidence.append("reports/tables/detector_neu.json")
    elif profile_id == "commutator":
        evidence.extend(
            [
                "reports/tables/segmenter_kolektor.json",
                "reports/tables/segmenter_kolektor_training.json",
                "reports/tables/ablation_segmenter_protocol.json",
            ]
        )
    else:
        evidence.append("reports/tables/anomaly_mvtec.json")
    if cpu_path.exists():
        evidence.append(str(cpu_path.relative_to(REPO)))

    return {
        "domain": config["domain"],
        "source": config["source"],
        "task": config["task"],
        "protocol": PROTOCOL[profile_id],
        "calibration": {
            "path": str(config["calibration"]),
            "rule": calibration["rule"],
            "alpha_fail": calibration["alpha_fail"],
            "alpha_review": calibration["alpha_review"],
            "sample_sizes": calibration["sample_sizes"],
        },
        "model_metrics": model_metrics,
        "operating_point": {
            "images": evaluation["images"],
            "defect_images": defect_n,
            "normal_images": normal_n,
            "verdict_counts": evaluation["verdict_counts"],
            "qc_review_count": evaluation.get("qc_review_count"),
            "review_rate": evaluation["review_rate"],
            "confusion": confusion,
            "defect_recall_non_pass": count_block(confusion["non_pass"]["detected"], defect_n),
            "defect_recall_fail": count_block(confusion["fail"]["detected"], defect_n),
            "normal_false_alarm_non_pass": count_block(
                confusion["non_pass"]["false_alarms"], normal_n
            ),
            "normal_false_alarm_fail": count_block(confusion["fail"]["false_alarms"], normal_n),
        },
        "latency": latency,
        "evidence": evidence,
    }


def advisory_summary() -> dict:
    table = load("advisory_audit.json")
    summary = {key: value for key, value in table.items() if key != "rows"}
    summary["evidence"] = "reports/tables/advisory_audit.json"
    return summary


def export_parity_summary() -> dict:
    table = load("export_parity.json")
    segmenter = [
        row["max_abs_logit_difference"]
        for row in table["profiles"]
        if "max_abs_logit_difference" in row
    ]
    anomaly = [
        row["max_abs_feature_difference"]
        for row in table["profiles"]
        if "max_abs_feature_difference" in row
    ]
    detector = [row for row in table["profiles"] if "max_box_difference_px" in row]
    return {
        "segmenter_max_abs_logit_difference": max(segmenter) if segmenter else None,
        "anomaly_max_abs_feature_difference": max(anomaly) if anomaly else None,
        "detector_max_box_difference_px": (
            max(row["max_box_difference_px"] for row in detector) if detector else None
        ),
        "detector_max_confidence_difference": (
            max(row["max_confidence_difference"] for row in detector) if detector else None
        ),
        "detector_images": detector[0]["images"] if detector else None,
        "evidence": "reports/tables/export_parity.json",
        "note": "ONNX Runtime CPUExecutionProvider compared with PyTorch CPU on local inputs; the reference station executes PyTorch.",
    }


def main() -> None:
    payload = {
        "schema": "veriqc.results/2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "cpu": _cpu_name(),
            "gpu": _gpu_name(),
            "system": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
            "software": {
                "torch": _package("torch"),
                "torchvision": _package("torchvision"),
                "ultralytics": _package("ultralytics"),
                "numpy": _package("numpy"),
                "executorch": _package("executorch"),
                "onnxruntime": _package("onnxruntime"),
            },
            "raspberry_pi": None,
        },
        "evaluation_protocol": {
            "split_discipline": (
                "Official test splits are held out from fitting, calibration and "
                "threshold selection. Segmenter preprocessing and protocol arms were "
                "compared on the official test split during development, so the "
                "selected arm is a development choice rather than a pristine one-shot "
                "holdout. The steel and Kolektor calibration/early-stopping splits also "
                "overlap; that limitation is disclosed and quantified by the protocol "
                "ablation."
            ),
            "quality_gate": (
                "Evaluation applies the same acquisition gate as the station before "
                "the decision. A failed gate returns REVIEW and never executes the models."
            ),
            "decision": (
                "Anomaly and dual profiles use split-conformal tail p-values; FAIL "
                "at alpha 0.05, REVIEW at alpha 0.10, with a missing or non-finite "
                "channel failing closed. The steel profile uses a recall-targeted "
                "screening threshold and makes no false-alarm claim."
            ),
        },
        "profiles": {profile_id: profile_record(profile_id) for profile_id in PROFILES},
        "ablations": {
            "anomaly_layer_budget": load("ablation_anomaly_layers.json"),
            "commutator_channel": load("ablation_commutator_channels.json"),
            "calibration_vs_fixed_threshold": load("ablation_calibration.json"),
            "segmenter_protocol": load("ablation_segmenter_protocol.json"),
            "anomaly_kolektor": load("ablation_anomaly_kolektor.json"),
        },
        "advisory": advisory_summary(),
        "artifact_manifests": sorted(
            str(path.relative_to(REPO)) for path in (REPO / "reports" / "models").glob("*.json")
        ),
        "runtime_artifact_manifest": (
            "configs/artifacts.json" if (REPO / "configs" / "artifacts.json").exists() else None
        ),
        "export_parity": export_parity_summary(),
        "not_measured": [
            "Raspberry Pi 5 latency, memory, thermal and co-resident advisory scheduling",
            "Human advisory usefulness rating",
            "Hailo NPU execution",
            "Normal-steel false-alarm rate (no normal NEU-DET population)",
        ],
    }
    output = REPO / "reports" / "results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    missing = [
        evidence
        for record in payload["profiles"].values()
        for evidence in record["evidence"]
        if not (REPO / evidence).exists()
    ]
    if missing:
        raise SystemExit(f"results index references missing evidence: {missing}")
    print(output.relative_to(REPO), "written")
    for profile_id, record in payload["profiles"].items():
        op = record["operating_point"]
        print(
            profile_id,
            op["verdict_counts"],
            "non-pass",
            op["defect_recall_non_pass"]["rate"] if op["defect_recall_non_pass"] else None,
            "fail",
            op["defect_recall_fail"]["rate"] if op["defect_recall_fail"] else None,
        )


if __name__ == "__main__":
    main()
