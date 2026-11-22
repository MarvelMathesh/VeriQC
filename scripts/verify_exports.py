#!/usr/bin/env python3
"""Verify the exported ONNX vision artifacts against the PyTorch models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import list_profiles, load_profile  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.models import PatchCore, Segmenter  # noqa: E402


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def detector_parity(profile, manifest: dict) -> dict:
    from ultralytics import YOLO

    export = manifest["artifacts"]["known"]
    source = profile.known.path
    onnx_path = REPO / export["path"]
    reference = YOLO(str(source))
    exported = YOLO(str(onnx_path))
    items = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")["items"]["test"][:8]
    box_diffs = []
    confidence_diffs = []
    counts = []
    for item in items:
        first = reference.predict(
            item["image"], imgsz=profile.known.input_size, conf=0.01, iou=0.5,
            device="cpu", verbose=False,
        )[0].boxes
        second = exported.predict(
            item["image"], imgsz=profile.known.input_size, conf=0.01, iou=0.5,
            device="cpu", verbose=False,
        )[0].boxes
        a_boxes = first.xyxy.cpu().numpy()
        b_boxes = second.xyxy.cpu().numpy()
        a_conf = first.conf.cpu().numpy()
        b_conf = second.conf.cpu().numpy()
        a_cls = first.cls.cpu().numpy()
        b_cls = second.cls.cpu().numpy()
        if len(a_boxes) != len(b_boxes):
            counts.append({"reference": len(a_boxes), "onnx": len(b_boxes)})
            continue
        if len(a_boxes) == 0:
            continue
        order_a = np.lexsort((a_conf, a_cls))
        order_b = np.lexsort((b_conf, b_cls))
        box_diffs.append(float(np.abs(a_boxes[order_a] - b_boxes[order_b]).max()))
        confidence_diffs.append(float(np.abs(a_conf[order_a] - b_conf[order_b]).max()))
    return {
        "artifact": export["path"],
        "images": len(items),
        "max_box_difference_px": max(box_diffs) if box_diffs else 0.0,
        "max_confidence_difference": max(confidence_diffs) if confidence_diffs else 0.0,
        "count_mismatches": counts,
    }


def segmenter_parity(profile, manifest: dict) -> dict:
    export = manifest["artifacts"]["known"]
    onnx_path = REPO / export["path"]
    segmenter = Segmenter.from_checkpoint(profile.known.path, device="cpu")
    height, width = segmenter.config.input_height, segmenter.config.input_width
    tensor = torch.randn(1, 3, height, width)
    with torch.no_grad():
        reference = segmenter.model(tensor).numpy()
    session = _session(onnx_path)
    output = session.run(None, {session.get_inputs()[0].name: tensor.numpy()})[0]
    return {
        "artifact": export["path"],
        "input_shape": [1, 3, height, width],
        "max_abs_logit_difference": float(np.abs(reference - output).max()),
    }


def anomaly_parity(profile, manifest: dict) -> dict:
    export = manifest["artifacts"]["anomaly"]
    onnx_path = REPO / export["backbone"]
    model = PatchCore.load(profile.anomaly.path, device="cpu")
    size = model.config.input_size
    tensor = torch.randn(1, 3, size, size)
    with torch.no_grad():
        reference = model.extractor(tensor)
    session = _session(onnx_path)
    feeds = {session.get_inputs()[0].name: tensor.numpy()}
    outputs = session.run(None, feeds)
    names = [output.name for output in session.get_outputs()]
    differences = {
        name: float(np.abs(reference[name].numpy() - value).max())
        for name, value in zip(names, outputs)
    }
    return {
        "artifact": export["backbone"],
        "input_shape": [1, 3, size, size],
        "max_abs_feature_difference": max(differences.values()),
        "per_layer": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="all")
    parser.add_argument("--out", default="reports/tables/export_parity.json")
    args = parser.parse_args()

    paths = list_profiles()
    if args.profile != "all":
        paths = [REPO / "configs" / "profiles" / f"{args.profile}.yaml"]
    rows = []
    for path in paths:
        profile = load_profile(path, validate=False)
        exported_manifest = REPO / "models" / "profiles" / profile.profile_id / "export" / "manifest.json"
        if not exported_manifest.exists():
            continue
        manifest = json.loads(exported_manifest.read_text())
        if profile.known is not None:
            if profile.known.kind == "detector":
                row = detector_parity(profile, manifest)
            else:
                row = segmenter_parity(profile, manifest)
        elif profile.anomaly.enabled:
            row = anomaly_parity(profile, manifest)
        else:
            continue
        row["profile_id"] = profile.profile_id
        rows.append(row)
        print(profile.profile_id, row, flush=True)

    payload = {
        "runtime": "ONNX Runtime CPUExecutionProvider vs PyTorch CPU",
        "note": "These exports are deployment artifacts; the reference station executes PyTorch.",
        "profiles": rows,
    }
    output = REPO / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output.relative_to(REPO))


if __name__ == "__main__":
    main()
