#!/usr/bin/env python3
"""Measure station-aligned vision latency and process memory for one profile."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import test_items  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    profile = load_profile(REPO / "configs" / "profiles" / f"{args.profile}.yaml")
    os.environ["VERIQC_THREADS"] = str(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))
    pipeline = Pipeline(profile)
    paths = [item["image"] for item in test_items(profile, limit=args.n + args.warmup)]
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
        "backend": pipeline.backend,
        "device": pipeline.device,
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
