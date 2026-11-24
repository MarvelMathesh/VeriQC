#!/usr/bin/env python3
"""Render publication-quality figures from the checked-in evidence tables.

Every number is read from a measured table or a per-image score dump. The
script does not estimate, smooth, or invent a value. Host and Raspberry Pi
measurements are kept visually separate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "results.json"
TABLES = REPO / "reports" / "tables"
OUTPUT = REPO / "reports" / "figures"

PROFILES = (
    "steel_surface",
    "commutator",
    "textile_carpet",
    "packaging_bottle",
    "pharma_capsule",
    "automotive_metal_nut",
    "electronics_transistor",
)
PRETTY = {
    "steel_surface": "Steel Surface",
    "commutator": "Commutator",
    "textile_carpet": "Carpet",
    "packaging_bottle": "Bottle",
    "pharma_capsule": "Capsule",
    "automotive_metal_nut": "Metal Nut",
    "electronics_transistor": "Transistor",
}
SCORE_PROFILES = PROFILES[1:]

# Cohesive publication palette
BLUE = "#255E8E"
AMBER = "#D96B27"
CRIMSON = "#C23B4D"
TEAL = "#2A7F62"
PLUM = "#6B4C9A"
CYAN = "#0E7490"

INK = "#0F172A"
TEXT = "#334155"
MUTED = "#64748B"
SPINE = "#CBD5E1"
GRID = "#E2E8F0"
BAND = "#F8FAFC"

CURVE_PALETTE = [BLUE, AMBER, CRIMSON, TEAL, PLUM, CYAN]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def _load(name: str) -> dict:
    return json.loads((TABLES / name).read_text())


def _style_axis(ax, grid_axis: str = "x") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)


def _save(fig, filename: str) -> None:
    fig.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )
    plt.close(fig)


def operating_points(data: dict) -> None:
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.9),
        gridspec_kw={"width_ratios": (1.04, 1.0), "wspace": 0.28},
    )
    positions = np.arange(len(PROFILES))
    for i in range(0, len(PROFILES), 2):
        left.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)

    series = (
        ("non-PASS", "defect_recall_non_pass", BLUE, "o", -0.14),
        ("FAIL", "defect_recall_fail", CRIMSON, "s", 0.14),
    )
    for label, key, color, marker, offset in series:
        values, lower, upper = [], [], []
        for profile in PROFILES:
            block = data["profiles"][profile]["operating_point"][key]
            values.append(block["rate"])
            lower.append(block["rate"] - block["ci95"][0])
            upper.append(block["ci95"][1] - block["rate"])
        left.errorbar(
            values,
            positions + offset,
            xerr=[lower, upper],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3.2,
            capthick=1.1,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )

    left.set_yticks(positions, [PRETTY[profile] for profile in PROFILES])
    left.invert_yaxis()
    left.set_ylim(len(PROFILES) - 0.5, -0.5)
    left.set_xlim(0.60, 1.03)
    left.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    left.set_xlabel("Recall on official test defects (95% CI)", labelpad=7)
    left.set_title("Defect recall at calibrated operating point", loc="left", pad=10)
    _style_axis(left, "x")

    normal_profiles = [
        profile
        for profile in PROFILES
        if data["profiles"][profile]["operating_point"]["normal_false_alarm_non_pass"]
    ]
    n_positions = np.arange(len(normal_profiles))
    for i in range(0, len(normal_profiles), 2):
        right.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)

    for label, key, color, marker, offset in (
        ("non-PASS", "normal_false_alarm_non_pass", BLUE, "o", -0.14),
        ("FAIL", "normal_false_alarm_fail", CRIMSON, "s", 0.14),
    ):
        values, lower, upper = [], [], []
        for profile in normal_profiles:
            block = data["profiles"][profile]["operating_point"][key]
            values.append(block["rate"])
            lower.append(block["rate"] - block["ci95"][0])
            upper.append(block["ci95"][1] - block["rate"])
        right.errorbar(
            values,
            n_positions + offset,
            xerr=[lower, upper],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3.2,
            capthick=1.1,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )

    right.set_yticks(n_positions, [PRETTY[profile] for profile in normal_profiles])
    right.invert_yaxis()
    right.set_ylim(len(normal_profiles) - 0.5, -0.5)
    right.set_xlim(-0.008, 0.36)
    right.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.set_xlabel("Normal-image false-alarm rate (95% CI)", labelpad=7)
    right.set_title("False alarms on official test normals", loc="left", pad=10)
    _style_axis(right, "x")

    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=8.5,
    )
    fig.suptitle(
        "Calibrated Operating Points After the Acquisition Gate",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.10, right=0.98)
    _save(fig, "operating_points.png")


def runtime_profile(data: dict) -> None:
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.9),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": 0.28},
    )
    positions = np.arange(len(PROFILES))
    height = 0.35
    host = [
        data["profiles"][profile]["latency"]["cpu_station_ms"]["median"]
        for profile in PROFILES
    ]
    pi = [
        data["pi_vision_profiles"][profile]["total_ms"]["median"]
        for profile in PROFILES
    ]
    throughput = [
        data["pi_vision_profiles"][profile]["throughput_images_per_second"]
        for profile in PROFILES
    ]

    left.barh(positions - height / 2, host, height=height, color=BLUE, label="Host CPU", zorder=3)
    left.barh(positions + height / 2, pi, height=height, color=AMBER, label="Raspberry Pi 5 CPU", zorder=3)
    left.set_yticks(positions, [PRETTY[profile] for profile in PROFILES])
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlim(14, 2800)
    ticks = [20, 50, 100, 200, 500, 1000, 2000]
    left.xaxis.set_major_locator(FixedLocator(ticks))
    left.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    left.xaxis.set_minor_locator(FixedLocator([]))
    left.set_xlabel("Median vision latency (ms, log scale)", labelpad=7)
    left.set_title("Vision: one ONNX channel per profile", loc="left", pad=10)
    left.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="upper right", fontsize=8.5)

    for pos, h_val, p_val, rate in zip(positions, host, pi, throughput):
        left.text(
            h_val * 1.07,
            pos - height / 2,
            f"{h_val:.0f} ms",
            va="center",
            fontsize=7.6,
            color=TEXT,
        )
        left.text(
            p_val * 1.07,
            pos + height / 2,
            f"{p_val:.0f} ms · {rate:.2f} img/s",
            va="center",
            fontsize=7.6,
            color=TEXT,
        )
    _style_axis(left, "x")

    host_advisory = data["advisory"]
    pi_advisory = data["pi_benchmark"]["advisory"]
    names = ["Host CPU", "Raspberry Pi 5"]
    ttft = [host_advisory["ttft_ms_median"] / 1000.0, pi_advisory["ttft_ms"]["median"] / 1000.0]
    total = [
        host_advisory["latency_ms_median"] / 1000.0,
        pi_advisory["latency_ms"]["median"] / 1000.0,
    ]
    decode = [total[i] - ttft[i] for i in range(2)]
    r_pos = np.arange(2)
    right.bar(r_pos, ttft, width=0.44, color=BLUE, label="Time to first token", zorder=3)
    right.bar(r_pos, decode, width=0.44, bottom=ttft, color=AMBER, label="Decode", zorder=3)
    right.set_xticks(r_pos, names)
    right.set_xlim(-0.55, 1.55)
    right.set_ylabel("Seconds per advisory", labelpad=7)
    right.set_title("Advisory: Llama 3.2 1B / ExecuTorch", loc="left", pad=10)
    right.set_ylim(0, max(total) * 1.28)
    right.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="upper left", fontsize=8.5)

    rates = [
        host_advisory["tokens_per_second_median"],
        pi_advisory["tokens_per_second_median"],
    ]
    accepted = [
        f"{host_advisory['accepted']}/{host_advisory['cases']}",
        f"{pi_advisory['accepted']}/{pi_advisory['cases']}",
    ]
    for idx, (val, rate, acc) in enumerate(zip(total, rates, accepted)):
        right.text(
            idx,
            val + 0.45,
            f"{val:.1f} s\n{rate:.2f} tok/s · {acc} accepted",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=TEXT,
            linespacing=1.3,
        )
    _style_axis(right, "y")

    fig.suptitle(
        "Measured CPU Runtime: Vision & Isolated Advisory",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.10, right=0.98)
    _save(fig, "runtime_profile.png")


def ablation_summary() -> None:
    layers = _load("ablation_anomaly_layers.json")["arms"]
    channels = _load("ablation_commutator_channels.json")["arms"]
    calibration = _load("ablation_calibration.json")["arms"]

    fig, (left, middle, right) = plt.subplots(
        1,
        3,
        figsize=(15.4, 4.9),
        gridspec_kw={"wspace": 0.25},
    )

    # Panel 1: PatchCore layer budget
    pos = np.arange(len(layers))
    width = 0.26
    metrics = (
        ("image_auroc", "Image AUROC", BLUE),
        ("pixel_auroc", "Pixel AUROC", AMBER),
        ("au_pro", "AU-PRO", TEAL),
    )
    for idx, (key, label, color) in enumerate(metrics):
        values = [row[key] for row in layers]
        left.bar(pos + (idx - 1) * width, values, width=width, color=color, label=label, zorder=3)
        for p, v in zip(pos, values):
            left.text(
                p + (idx - 1) * width,
                v + 0.003,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=TEXT,
            )
    left.set_xticks(pos, [row["arm"] for row in layers])
    left.set_ylim(0.84, 1.052)
    left.set_ylabel("Score", labelpad=6)
    left.set_title("PatchCore layer budget (Carpet)", loc="left", pad=10)
    left.legend(
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=7.6,
        loc="upper right",
        ncol=1,
    )
    _style_axis(left, "y")

    # Panel 2: Kolektor channel ablation
    pos = np.arange(len(channels))
    width = 0.35
    recall = [row["recall"] for row in channels]
    false_alarm = [row["false_alarm_rate"] for row in channels]
    middle.bar(pos - width / 2, recall, width=width, color=BLUE, label="Recall", zorder=3)
    middle.bar(pos + width / 2, false_alarm, width=width, color=CRIMSON, label="False alarm", zorder=3)
    middle.set_xticks(pos, [row["arm"].replace("-", "\n") for row in channels])
    middle.set_ylim(0, 1.22)
    middle.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    middle.set_title("Kolektor channel ablation (Fold 0)", loc="left", pad=10)
    middle.legend(
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=7.8,
        loc="upper center",
        ncol=2,
    )
    for p, v in zip(pos, recall):
        middle.text(p - width / 2, v + 0.02, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    for p, v in zip(pos, false_alarm):
        middle.text(p + width / 2, v + 0.02, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    _style_axis(middle, "y")

    # Panel 3: Calibration ablation
    pos = np.arange(len(calibration))
    width = 0.35
    fixed = [row["fixed_0.5"]["false_alarm_rate"] for row in calibration]
    calibrated = [row["calibrated"]["false_alarm_rate"] for row in calibration]
    right.bar(pos - width / 2, fixed, width=width, color=CRIMSON, label="Fixed 0.5", zorder=3)
    right.bar(pos + width / 2, calibrated, width=width, color=BLUE, label="Calibrated", zorder=3)
    right.set_xticks(pos, [PRETTY[row["profile_id"]] for row in calibration])
    right.set_ylim(0, 0.132)
    right.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.set_title("Calibration: normal false alarms at FAIL", loc="left", pad=10)
    right.legend(
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        fontsize=7.8,
        loc="upper left",
    )
    for p, v in zip(pos, fixed):
        right.text(p - width / 2, v + 0.0025, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    for p, v in zip(pos, calibrated):
        right.text(p + width / 2, v + 0.0025, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    _style_axis(right, "y")

    fig.suptitle(
        "Design Ablations: Single Channel, Layer Budget & Threshold Calibration",
        x=0.03,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.05, right=0.985)
    _save(fig, "ablation_summary.png")


def score_curves() -> None:
    available = []
    for profile in SCORE_PROFILES:
        path = TABLES / "scores" / f"{profile}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = [
            row
            for row in payload["rows"]
            if row.get("qc_ok", True) and row.get(payload["score_kind"]) is not None
        ]
        labels = np.asarray([bool(row["label"]) for row in rows], dtype=int)
        scores = np.asarray([float(row[payload["score_kind"]]) for row in rows], dtype=float)
        if labels.min() == labels.max():
            continue
        available.append((profile, labels, scores))
    if not available:
        print("no score dumps found; run scripts/evaluate.py --dump-scores first")
        return

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.0),
        gridspec_kw={"wspace": 0.22},
    )
    for color, (profile, labels, scores) in zip(CURVE_PALETTE, available):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        left.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.0,
            alpha=0.92,
            label=f"{PRETTY[profile]} (AUC {auc:.3f})",
            zorder=3,
        )
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        right.plot(
            recall,
            precision,
            color=color,
            linewidth=2.0,
            alpha=0.92,
            label=f"{PRETTY[profile]} (AP {ap:.3f})",
            zorder=3,
        )

    left.plot([0, 1], [0, 1], color="#94A3B8", linewidth=1.0, linestyle="--", zorder=2)
    left.set_xlim(-0.02, 1.02)
    left.set_ylim(-0.02, 1.03)
    left.set_xlabel("False-positive rate", labelpad=7)
    left.set_ylabel("True-positive rate", labelpad=7)
    left.set_title("Image-level ROC on official test split", loc="left", pad=10)
    left.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.0, loc="lower right")
    _style_axis(left, "both")

    right.set_xlim(-0.02, 1.02)
    right.set_ylim(-0.02, 1.03)
    right.set_xlabel("Recall", labelpad=7)
    right.set_ylabel("Precision", labelpad=7)
    right.set_title("Image-level precision-recall", loc="left", pad=10)
    right.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.0, loc="lower left")
    _style_axis(right, "both")

    fig.suptitle(
        "Threshold-Free Ranking Quality of the Production ONNX Channel",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.06, right=0.98)
    _save(fig, "roc_pr.png")


def design_selection() -> None:
    archive = TABLES / "archive" / "design_bakeoffs"
    detector = json.loads((archive / "detector_bakeoff.json").read_text())["arms"]
    backbone = json.loads((archive / "backbone_pareto.json").read_text())["arms"]
    sweep = json.loads((archive / "anomaly_sweep.json").read_text())["arms"]
    cascade = json.loads((archive / "cascade_bakeoff.json").read_text())["domains"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.4, 8.6),
        gridspec_kw={"hspace": 0.34, "wspace": 0.24},
    )
    left, right = axes[0]
    lower_left, lower_right = axes[1]

    # Top-left: Detector bakeoff
    pos = np.arange(len(detector))
    width = 0.35
    map50 = [row["map50"] for row in detector]
    map5095 = [row["map50_95"] for row in detector]
    left.bar(pos - width / 2, map50, width=width, color=BLUE, label="mAP50", zorder=3)
    left.bar(pos + width / 2, map5095, width=width, color=AMBER, label="mAP50-95", zorder=3)
    left.set_xticks(pos, [row["id"] for row in detector])
    left.set_ylim(0, 0.88)
    left.set_ylabel("Score", labelpad=6)
    left.set_title("Detector family and input size (Kolektor validation)", loc="left", pad=10)
    left.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8.2, loc="upper right")
    for p, v in zip(pos, map50):
        left.text(p - width / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, color=TEXT)
    for p, v in zip(pos, map5095):
        left.text(p + width / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, color=TEXT)
    _style_axis(left, "y")

    # Top-right: Backbone Pareto
    for row in backbone:
        within = row["within_memory_budget"] and row["within_parameter_budget"]
        color = TEAL if within else CRIMSON
        right.scatter(
            row["cpu_latency_ms_4_threads"],
            row["mean_image_auroc"],
            s=max(55.0, row["memory_bank_mb"] * 85.0),
            color=color,
            alpha=0.80,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        name = row["backbone"].replace("_", " ")
        if row["backbone"] == "wide_resnet50_2":
            offset, ha = (-18, -2), "right"
        elif row["backbone"] == "mobilenet_v3_small":
            offset, ha = (8, 5), "left"
        else:
            offset, ha = (9, 5), "left"
        right.annotate(
            name,
            (row["cpu_latency_ms_4_threads"], row["mean_image_auroc"]),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            va="center",
            fontsize=8.0,
            color=TEXT,
        )
    right.set_xscale("log")
    right.set_xlim(0.7, 180)
    right.set_ylim(0.51, 0.97)
    right.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    right.xaxis.set_minor_locator(FixedLocator([]))
    right.set_xlabel("CPU latency, 4 threads (ms, log scale)", labelpad=6)
    right.set_ylabel("Mean image AUROC", labelpad=6)
    right.set_title("Backbone Pareto under budget screen", loc="left", pad=10)
    right.scatter([], [], s=60, color=TEAL, label="Within budget")
    right.scatter([], [], s=60, color=CRIMSON, label="Outside budget (bubble = bank MB)")
    right.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=7.8, loc="lower right")
    _style_axis(right, "both")

    # Bottom-left: Input, resize & bank sweep
    sweep_rows = []
    for row in sweep:
        config = row["config"]
        b_name = str(config["backbone"]).replace("_", " ")
        mode = "LB" if config["resize_mode"] == "letterbox" else str(config["resize_mode"])
        bank = int(config["bank_size"])
        bank_label = f"{bank // 1000}k" if bank >= 1000 else str(bank)
        label = (
            f"{config['input_size']} {mode} {bank_label}"
            if b_name == "efficientnet b0"
            else f"{b_name} {config['input_size']}"
        )
        sweep_rows.append((label, float(row["image_auroc"]), float(row["detection_rate_at_threshold"])))

    pos = np.arange(len(sweep_rows))
    lower_left.barh(
        pos,
        [row[1] for row in sweep_rows],
        color=[BLUE if row[1] >= 0.80 else AMBER for row in sweep_rows],
        height=0.62,
        zorder=3,
    )
    lower_left.set_yticks(pos, [row[0] for row in sweep_rows])
    lower_left.invert_yaxis()
    lower_left.set_xlim(0.50, 0.95)
    lower_left.set_xlabel("Image AUROC at α = 0.05", labelpad=6)
    lower_left.set_title("Input, resize, and memory bank sweep", loc="left", pad=10)
    for p, (_, auroc, detected) in zip(pos, sweep_rows):
        lower_left.text(
            auroc + 0.007,
            p,
            f"{auroc:.3f} · det {detected:.2f}",
            va="center",
            fontsize=7.6,
            color=TEXT,
        )
    _style_axis(lower_left, "x")

    # Bottom-right: Gated cascade vs production
    pos = np.arange(2)
    width = 0.34
    production = [
        cascade[0]["production"]["known_defect_nonpass_recall"],
        cascade[1]["production"]["novel_defect_nonpass_recall"],
    ]
    gated = [
        cascade[0]["cascade_gated"]["known_defect_nonpass_recall"],
        cascade[1]["cascade_gated"]["novel_defect_nonpass_recall"],
    ]
    lower_right.bar(pos - width / 2, production, width=width, color=BLUE, label="Production (single channel)", zorder=3)
    lower_right.bar(pos + width / 2, gated, width=width, color=AMBER, label="Gated cascade", zorder=3)
    lower_right.set_xticks(pos, ["Kolektor\nknown defect", "MVTec\nnovel defect"])
    lower_right.set_ylim(0, 1.22)
    lower_right.set_ylabel("non-PASS recall", labelpad=6)
    lower_right.set_title("Gated cascade vs. production", loc="left", pad=10)
    lower_right.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=7.8, loc="upper right")
    execution = [
        (cascade[0]["production"]["detector_execution_rate"], cascade[0]["cascade_gated"]["detector_execution_rate"]),
        (cascade[1]["production"]["detector_execution_rate"], cascade[1]["cascade_gated"]["detector_execution_rate"]),
    ]
    for idx, (prod, gate) in enumerate(zip(production, gated)):
        lower_right.text(
            idx - width / 2,
            prod + 0.025,
            f"{prod:.3f}\n({execution[idx][0] * 100:.0f}% calls)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=TEXT,
            linespacing=1.25,
        )
        lower_right.text(
            idx + width / 2,
            gate + 0.025,
            f"{gate:.3f}\n({execution[idx][1] * 100:.1f}% calls)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=TEXT,
            linespacing=1.25,
        )
    _style_axis(lower_right, "y")

    fig.suptitle(
        "Architecture Bake-Offs: Detector, Backbone, Input & Cascade Selection",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.11, right=0.98)
    _save(fig, "design_selection.png")


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"missing results index: {RESULTS}")
    _configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text())
    operating_points(data)
    runtime_profile(data)
    ablation_summary()
    score_curves()
    design_selection()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path.relative_to(REPO))


if __name__ == "__main__":
    sys.exit(main())
