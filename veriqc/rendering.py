"""Overlay rendering for the operator dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from veriqc import REPO_ROOT
from veriqc.inspection import InspectionResult

__all__ = ["save_overlay"]


def _prune_overlays(directory: Path, keep: int) -> None:
    """Bound the runtime overlay cache; old records keep their audit data."""

    if keep <= 0:
        return
    files = sorted(
        directory.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in files[:-keep]:
        try:
            path.unlink()
        except OSError:
            pass


def save_overlay(
    image: Image.Image,
    result: InspectionResult,
    heatmap: np.ndarray | None,
    known_mask: np.ndarray | None,
    *,
    overlay_root: Path,
    known_threshold: float,
) -> str | None:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if heatmap is not None and result.verdict != "PASS":
        normalized = heatmap.astype(np.float32)
        high = float(normalized.max())
        if high > 0:
            normalized = normalized / high
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        canvas = cv2.addWeighted(canvas, 0.65, colored, 0.35, 0)
    if known_mask is not None and result.verdict != "PASS":
        binary = (known_mask >= known_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    for finding in result.findings:
        if finding.bbox is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in finding.bbox)
        color = (0, 200, 255) if finding.source == "known" else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{finding.defect_class} {finding.severity or ''}".strip()
        cv2.putText(
            canvas,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    verdict = result.verdict
    color = (
        (0, 200, 0)
        if verdict == "PASS"
        else (0, 0, 255)
        if verdict == "FAIL"
        else (0, 200, 255)
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 36), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"{verdict} | {result.profile_id} | {result.part_id}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    target = overlay_root / "overlays" / f"{result.inspection_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        return None
    try:
        keep = int(os.environ.get("VERIQC_OVERLAY_KEEP", "500"))
    except ValueError:
        keep = 500
    _prune_overlays(target.parent, keep)
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)
