#!/usr/bin/env python3
"""Prepare deterministic split manifests for NEU-DET, KolektorSDD and MVTec AD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.datasets import prepare_kolektor, prepare_mvtec, prepare_neu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["neu", "kolektor", "mvtec", "all"], default="all", nargs="?")
    args = parser.parse_args()

    if args.source in ("neu", "all"):
        manifest = prepare_neu()
        print("NEU:", manifest["counts"])
    if args.source in ("kolektor", "all"):
        manifest = prepare_kolektor()
        print(
            "Kolektor folds:",
            [fold["counts"] for fold in manifest["folds"]],
        )
    if args.source in ("mvtec", "all"):
        manifest = prepare_mvtec()
        for category, payload in manifest["categories"].items():
            print(category, payload["counts"])


if __name__ == "__main__":
    main()
