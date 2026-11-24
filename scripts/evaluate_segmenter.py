#!/usr/bin/env python3
"""Evaluate the saved Kolektor fold checkpoints on the official test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.metrics import au_pro, average_precision, best_f1, image_auroc, pixel_auroc  # noqa: E402
from veriqc.models.torch_segmenter import Segmenter  # noqa: E402

METRIC_SIZE = (512, 1408)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def evaluate_fold(segmenter: Segmenter, items: list[dict]) -> dict:
    scores = []
    labels = []
    maps = []
    masks = []
    for item in items:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            probability, _ = segmenter.predict_proba(rgb)
            size = rgb.size
        if item.get("label"):
            with Image.open(item["label"]) as mask:
                native_mask = np.asarray(mask.convert("L")) > 127
        else:
            native_mask = np.zeros((size[1], size[0]), dtype=bool)
        maps.append(cv2.resize(probability.astype(np.float32), METRIC_SIZE, interpolation=cv2.INTER_LINEAR))
        masks.append(
            cv2.resize(native_mask.astype(np.uint8), METRIC_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        )
        scores.append(float(probability.max()))
        labels.append(bool(item["is_defect"]))

    maps_array = np.stack(maps)
    masks_array = np.stack(masks)
    scores_array = np.asarray(scores)
    labels_array = np.asarray(labels)
    f1, precision, recall = best_f1(scores_array, labels_array)
    return {
        "images": len(items),
        "defect_images": int(labels_array.sum()),
        "image_auroc": round(image_auroc(scores_array, labels_array), 4),
        "pixel_auroc": round(pixel_auroc(maps_array, masks_array), 4),
        "pixel_ap": round(average_precision(maps_array, masks_array), 4),
        "au_pro": round(au_pro(masks_array, maps_array), 4),
        "best_f1": round(f1, 4),
        "best_precision": round(precision, 4),
        "best_recall": round(recall, 4),
        "metric_size": list(METRIC_SIZE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="models/profiles/commutator")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--out", default="reports/tables/segmenter_kolektor.json")
    args = parser.parse_args()

    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO / checkpoint_dir
    fold_indices = args.folds if args.folds is not None else list(range(len(manifest["folds"])))
    folds = []
    for fold_index in fold_indices:
        fold = manifest["folds"][fold_index]
        checkpoint = checkpoint_dir / f"fold{fold_index}_unet.pt"
        segmenter = Segmenter.from_checkpoint(checkpoint, device=args.device)
        metrics = evaluate_fold(segmenter, fold["test"])
        metrics.update(
            {
                "fold": fold_index,
                "input_size": [segmenter.config.input_width, segmenter.config.input_height],
                "weights": display_path(checkpoint),
                "weights_sha256": sha256_file(checkpoint),
            }
        )
        folds.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    mean = {
        key: round(sum(row[key] for row in folds) / len(folds), 4)
        for key in ("image_auroc", "pixel_auroc", "pixel_ap", "au_pro", "best_f1")
    }
    payload = {"folds": folds, "mean": mean, "metric_size": list(METRIC_SIZE)}
    output = REPO / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
