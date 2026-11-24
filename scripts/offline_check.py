#!/usr/bin/env python3
"""Verify that an inspection runs with outbound networking disabled."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import test_items  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402
from veriqc.station.dashboard import DASHBOARD_HTML  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="commutator")
    parser.add_argument("--image", default=None)
    parser.add_argument("--out", default="reports/tables/offline_check.json")
    args = parser.parse_args()

    profile = load_profile(REPO / "configs" / "profiles" / f"{args.profile}.yaml")
    image_path = Path(args.image) if args.image else Path(test_items(profile, limit=1)[0]["image"])
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    from PIL import Image

    pipeline = Pipeline(profile)

    original_socket = socket.socket
    original_create = socket.create_connection

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled during the offline check")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    started = time.perf_counter()
    try:
        with Image.open(image_path) as opened:
            result = pipeline.analyze(opened.convert("RGB"), "OFFLINE-CHECK")
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_create  # type: ignore[assignment]

    external_links = [
        token for token in ("http://", "https://", "//cdn", "unpkg", "jsdelivr")
        if token in DASHBOARD_HTML
    ]
    payload = {
        "ok": not external_links,
        "profile_id": args.profile,
        "backend": pipeline.backend,
        "image": str(image_path),
        "verdict": result.result.verdict,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "external_dashboard_assets": external_links,
        "network_blocked": True,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output = REPO / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit("dashboard contains an external asset reference")


if __name__ == "__main__":
    main()
