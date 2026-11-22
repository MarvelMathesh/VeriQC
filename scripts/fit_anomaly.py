#!/usr/bin/env python3
"""Fit and evaluate the normal-only PatchCore channel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.datasets import load_manifest, split_by_part  # noqa: E402
from veriqc.metrics import au_pro, average_precision, image_auroc, pixel_auroc  # noqa: E402
from veriqc.models.patchcore import PatchCore, PatchCoreConfig  # noqa: E402

MVTEC_PROFILES = {
    "carpet": "textile_carpet",
    "bottle": "packaging_bottle",
    "capsule": "pharma_capsule",
    "metal_nut": "automotive_metal_nut",
    "transistor": "electronics_transistor",
}


def evaluate(model: PatchCore, items: list[dict], metric_size: tuple[int, int] = (512, 512)) -> dict:
    scores = []
    labels = []
    maps = []
    masks = []
    heatmap_latency = []
    for item in items:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            result = model.score(rgb)
            score = result.image_score
            heatmap = result.heatmap
            size = rgb.size
        scores.append(score)
        labels.append(bool(item["is_defect"]))
        if item.get("label"):
            with Image.open(item["label"]) as mask:
                native_mask = np.asarray(mask.convert("L")) > 127
        else:
            native_mask = np.zeros((size[1], size[0]), dtype=bool)
        masks.append(cv2.resize(native_mask.astype(np.uint8), metric_size, interpolation=cv2.INTER_NEAREST) > 0)
        maps.append(cv2.resize(heatmap.astype(np.float32), metric_size, interpolation=cv2.INTER_LINEAR))
        heatmap_latency.append(result.latency_ms)
    maps_array = np.stack(maps)
    masks_array = np.stack(masks)
    scores_array = np.asarray(scores)
    labels_array = np.asarray(labels)
    return {
        "images": len(items),
        "defect_images": int(labels_array.sum()),
        "image_auroc": round(image_auroc(scores_array, labels_array), 4),
        "pixel_auroc": round(pixel_auroc(maps_array, masks_array), 4),
        "pixel_ap": round(average_precision(maps_array, masks_array), 4),
        "au_pro": round(au_pro(masks_array, maps_array), 4),
        "score_ms_median": round(float(np.median(heatmap_latency)), 2),
    }


def normal_scores(model: PatchCore, items: list[dict]) -> list[float]:
    out = []
    for item in items:
        with Image.open(item["image"]) as image:
            out.append(model.score(image.convert("RGB")).image_score)
    return out


def fit_mvtec(categories: list[str], device: str) -> list[dict]:
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    reports = []
    for category in categories:
        payload = manifest["categories"].get(category)
        if payload is None:
            print(f"skip {category}: not in manifest")
            continue
        profile_id = MVTEC_PROFILES.get(category, category)
        out = REPO / "models" / "profiles" / profile_id / "patchcore.pt"
        fit_items = [item for item in payload["items"] if item["split"] == "fit"]
        cal_items = [item for item in payload["items"] if item["split"] == "cal"]
        test_items = [item for item in payload["items"] if item["split"] == "test"]
        started = time.time()
        model = PatchCore(PatchCoreConfig(bank_size=4000, coreset_pool=20000), device=device)
        model.fit([item["image"] for item in fit_items])
        model.save(out)
        test = evaluate(model, test_items, metric_size=(512, 512))
        cal_scores = normal_scores(model, cal_items)
        report = {
            "profile_id": profile_id,
            "source": f"mvtec:{category}",
            "fit_images": len(fit_items),
            "cal_images": len(cal_items),
            "weights": str(out.relative_to(REPO)),
            "weights_sha256": sha256_file(out),
            "bank_size": int(model.bank_size),
            "test": test,
            "cal_scores": cal_scores,
            "fit_seconds": round(time.time() - started, 1),
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    return reports


def fit_kolektor(folds: list[int], device: str) -> list[dict]:
    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    reports = []
    for fold_index in folds:
        fold = manifest["folds"][fold_index]
        train_normals = [item for item in fold["train"] if not item["is_defect"]]
        fit_normals, cal_normals = split_by_part(train_normals, 20261121 + fold_index)
        out = REPO / "models" / "profiles" / "commutator" / f"patchcore_fold{fold_index}.pt"
        if fold_index == 0:
            out = REPO / "models" / "profiles" / "commutator" / "patchcore.pt"
        started = time.time()
        model = PatchCore(PatchCoreConfig(bank_size=4000, coreset_pool=20000), device=device)
        model.fit([item["image"] for item in fit_normals])
        model.save(out)
        test = evaluate(model, fold["test"], metric_size=(512, 1408))
        cal_scores = normal_scores(model, cal_normals)
        report = {
            "profile_id": "commutator",
            "source": "kolektor-sdd",
            "fold": fold_index,
            "fit_images": len(fit_normals),
            "cal_images": len(cal_scores),
            "weights": str(out.relative_to(REPO)),
            "weights_sha256": sha256_file(out),
            "bank_size": int(model.bank_size),
            "test": test,
            "cal_scores": cal_scores,
            "fit_seconds": round(time.time() - started, 1),
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["mvtec", "kolektor", "all"])
    parser.add_argument("--categories", nargs="*", default=["carpet", "bottle", "capsule", "metal_nut", "transistor"])
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2])
    import torch

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    reports = []
    if args.source in ("mvtec", "all"):
        reports.extend(fit_mvtec(args.categories, args.device))
    if args.source in ("kolektor", "all"):
        reports.extend(fit_kolektor(args.folds, args.device))
    default_out = {
        "mvtec": "reports/tables/anomaly_mvtec.json",
        "kolektor": "reports/tables/ablation_anomaly_kolektor.json",
        "all": "reports/tables/anomaly_all.json",
    }[args.source]
    target = REPO / (args.out or default_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"reports": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
