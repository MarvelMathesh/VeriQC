#!/usr/bin/env python3
"""Measure station-aligned vision latency and process memory for one profile."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402

MVTEC_PROFILES = {
    "textile_carpet": "carpet",
    "packaging_bottle": "bottle",
    "pharma_capsule": "capsule",
    "automotive_metal_nut": "metal_nut",
    "electronics_transistor": "transistor",
}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sample(profile: str, n: int) -> list[str]:
    if profile == "steel_surface":
        manifest = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")
        return [item["image"] for item in manifest["items"]["test"][:n]]
    if profile == "commutator":
        manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
        return [item["image"] for item in manifest["folds"][0]["test"][:n]]
    category = MVTEC_PROFILES[profile]
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    return [
        item["image"]
        for item in manifest["categories"][category]["items"]
        if item["split"] == "test"
    ][:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))
    profile = load_profile(REPO / "configs" / "profiles" / f"{args.profile}.yaml")
    pipeline = Pipeline(profile, device=args.device)
    paths = sample(args.profile, args.n + args.warmup)
    if len(paths) <= args.warmup:
        raise SystemExit("not enough test images for the requested warmup")
    warmup, measured = paths[: args.warmup], paths[args.warmup :]

    for path in warmup:
        with Image.open(path) as image:
            pipeline.inspect(image.convert("RGB"), "WARMUP", overlay=False)

    totals: list[float] = []
    for path in measured:
        with Image.open(path) as image:
            result = pipeline.inspect(image.convert("RGB"), "MEASURE", overlay=False)
        totals.append(float(result.latency_ms["total_ms"]))

    payload = {
        "profile_id": args.profile,
        "device": args.device,
        "threads": args.threads,
        "warmup_images": len(warmup),
        "images": len(measured),
        "total_ms": {
            "median": round(statistics.median(totals), 2),
            "p95": round(float(np.quantile(totals, 0.95)), 2),
            "p99": round(float(np.quantile(totals, 0.99)), 2),
            "min": round(min(totals), 2),
            "max": round(max(totals), 2),
        },
        "peak_rss_mb": round(rss_mb(), 1),
        "artifact_sizes_bytes": {
            name: path.stat().st_size for name, path in profile.artifact_paths().items()
        },
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output = REPO / (args.out or f"reports/tables/latency_{args.profile}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
