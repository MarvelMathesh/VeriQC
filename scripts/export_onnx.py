#!/usr/bin/env python3
"""Export a profile's vision models to ONNX and its anomaly banks to NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.config import list_profiles, load_profile  # noqa: E402
from veriqc.models import Detector, PatchCore, Segmenter  # noqa: E402


def export_profile(path: Path, device: str) -> None:
    profile = load_profile(path)
    out = REPO / "models" / "profiles" / profile.profile_id / "export"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"profile_id": profile.profile_id, "artifacts": {}}
    if profile.known is not None:
        if profile.known.kind == "detector":
            detector = Detector.from_weights(
                profile.known.path,
                profile.known.classes,
                input_size=int(profile.known.input_size or 640),
                device=device,
            )
            target = out / "detector.onnx"
            detector.export_onnx(target)
        else:
            segmenter = Segmenter.from_checkpoint(profile.known.path, device=device)
            target = out / "segmenter.onnx"
            segmenter.export_onnx(target)
        manifest["artifacts"]["known"] = {
            "path": str(target.relative_to(REPO)),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
        data_file = target.with_suffix(target.suffix + ".data")
        if data_file.exists():
            manifest["artifacts"]["known"]["external_data"] = {
                "path": str(data_file.relative_to(REPO)),
                "sha256": sha256_file(data_file),
                "bytes": data_file.stat().st_size,
            }
    if profile.anomaly.enabled and profile.anomaly.path is not None:
        model = PatchCore.load(profile.anomaly.path, device=device)
        backbone = out / "anomaly_backbone.onnx"
        model.export_backbone_onnx(backbone)
        banks = out / "anomaly_banks.npz"
        np.savez(
            banks,
            image_bank=model.image_bank.numpy(),
            pixel_bank=model.pixel_bank.numpy(),
            image_layer=model.config.image_layer,
            pixel_layer=model.config.pixel_layer,
            input_size=model.config.input_size,
        )
        manifest["artifacts"]["anomaly"] = {
            "backbone": str(backbone.relative_to(REPO)),
            "backbone_sha256": sha256_file(backbone),
            "banks": str(banks.relative_to(REPO)),
            "banks_sha256": sha256_file(banks),
            "bank_size": model.bank_size,
        }
        data_file = backbone.with_suffix(backbone.suffix + ".data")
        if data_file.exists():
            manifest["artifacts"]["anomaly"]["external_data"] = {
                "path": str(data_file.relative_to(REPO)),
                "sha256": sha256_file(data_file),
                "bytes": data_file.stat().st_size,
            }
    target = out / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    published = REPO / "reports" / "models" / f"export_{profile.profile_id}.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    paths = list_profiles() if args.profile == "all" else [
        REPO / "configs" / "profiles" / f"{args.profile}.yaml"
    ]
    for path in paths:
        export_profile(path, args.device)


if __name__ == "__main__":
    main()
