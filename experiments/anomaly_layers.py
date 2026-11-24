#!/usr/bin/env python3
"""Layer-budget ablation for the anomaly channel (MVTec carpet)."""

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

from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.metrics import au_pro, image_auroc, pixel_auroc  # noqa: E402
from veriqc.models.torch_patchcore import PatchCore, PatchCoreConfig  # noqa: E402


def evaluate(model: PatchCore, items: list[dict]) -> dict:
    scores, labels, maps, masks = [], [], [], []
    for item in items:
        with Image.open(item["image"]) as image:
            rgb = image.convert("RGB")
            result = model.score(rgb)
        scores.append(result.image_score)
        labels.append(item["is_defect"])
        maps.append(cv2.resize(result.heatmap, (512, 512), interpolation=cv2.INTER_LINEAR))
        if item.get("label"):
            with Image.open(item["label"]) as mask:
                native = np.asarray(mask.convert("L")) > 127
        else:
            native = np.zeros((rgb.height, rgb.width), dtype=bool)
        masks.append(cv2.resize(native.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0)
    maps_array = np.stack(maps)
    masks_array = np.stack(masks)
    return {
        "image_auroc": round(image_auroc(np.asarray(scores), np.asarray(labels)), 4),
        "pixel_auroc": round(pixel_auroc(maps_array, masks_array), 4),
        "au_pro": round(au_pro(masks_array, maps_array), 4),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    payload = manifest["categories"]["carpet"]
    fit = [item["image"] for item in payload["items"] if item["split"] == "fit"]
    test = [item for item in payload["items"] if item["split"] == "test"]
    arms = []
    for name, image_layer, pixel_layer in (
        ("layer2-only", "layer2", "layer2"),
        ("layer3-only", "layer3", "layer3"),
        ("dual-layer", "layer2", "layer3"),
    ):
        model = PatchCore(
            PatchCoreConfig(image_layer=image_layer, pixel_layer=pixel_layer, bank_size=4000),
            device=args.device,
        )
        model.fit(fit)
        metrics = evaluate(model, test)
        arms.append({"arm": name, "image_layer": image_layer, "pixel_layer": pixel_layer, "bank_size": model.bank_size, **metrics})
        print(arms[-1], flush=True)
    output = REPO / "reports" / "tables" / "ablation_anomaly_layers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": "mvtec:carpet", "arms": arms}, indent=2) + "\n")


if __name__ == "__main__":
    main()
