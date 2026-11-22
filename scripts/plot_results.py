#!/usr/bin/env python3
"""Render the compact results overview from the measured index."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "results.json"
OUTPUT = REPO / "reports" / "figures" / "results_overview.png"

ANOMALY = (
    "textile_carpet",
    "packaging_bottle",
    "pharma_capsule",
    "automotive_metal_nut",
    "electronics_transistor",
)
LATENCY = ("steel_surface", "textile_carpet", "commutator")


def pretty(name: str) -> str:
    return name.replace("_", " ")


def main() -> None:
    data = json.loads(RESULTS.read_text())
    profiles = data["profiles"]

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.2),
        gridspec_kw={"width_ratios": (1.15, 1.0)},
    )
    fig.patch.set_facecolor("white")

    names = [pretty(name) for name in ANOMALY]
    values = [profiles[name]["model_metrics"]["image_auroc"] for name in ANOMALY]
    positions = np.arange(len(names))
    left.barh(positions, values, color="#2f5d8a", height=0.58)
    left.set_yticks(positions, names)
    left.invert_yaxis()
    left.set_xlim(0.90, 1.0)
    left.set_xlabel("image AUROC")
    left.set_title("MVTec AD anomaly profiles", loc="left", fontsize=11)
    for position, value in zip(positions, values):
        left.text(value + 0.0015, position, f"{value:.4f}", va="center", fontsize=8.5)
    left.grid(axis="x", color="#d7dde5", linewidth=0.7)
    left.set_axisbelow(True)
    for side in ("top", "right"):
        left.spines[side].set_visible(False)

    positions = np.arange(len(LATENCY))
    width = 0.36
    medians = [profiles[name]["latency"]["cpu_station_ms"]["median"] for name in LATENCY]
    p95 = [profiles[name]["latency"]["cpu_station_ms"]["p95"] for name in LATENCY]
    right.barh(positions - width / 2, medians, height=width, color="#2f5d8a", label="median")
    right.barh(positions + width / 2, p95, height=width, color="#9db4cf", label="p95")
    right.set_yticks(positions, [pretty(name) for name in LATENCY])
    right.invert_yaxis()
    right.set_xlabel("milliseconds per frame (4 CPU threads)")
    right.set_title("Host station vision path: median / p95", loc="left", fontsize=11)
    right.set_xlim(0, max(p95) * 1.22)
    for position, median, tail in zip(positions, medians, p95):
        right.text(median + 4, position - width / 2, f"{median:.0f}", va="center", fontsize=8)
        right.text(tail + 4, position + width / 2, f"{tail:.0f}", va="center", fontsize=8)
    right.grid(axis="x", color="#d7dde5", linewidth=0.7)
    right.set_axisbelow(True)
    for side in ("top", "right"):
        right.spines[side].set_visible(False)

    fig.text(
        0.01,
        0.01,
        "Measured on the development host (i7-13700HX, 4 CPU threads). No Raspberry Pi measurement is claimed.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    print(OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
