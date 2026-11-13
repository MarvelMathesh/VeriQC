"""Production detector interface: Ultralytics YOLO26n.

VeriQC uses a single unified detector at the edge (YOLO26 nano).
Trained on the unified 7-class taxonomy (NEU-DET 6 classes + KolektorSDD crack).
Supports PyTorch weights (.pt) and ONNX export for edge runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from veriqc.metrics.detection import Detection

__all__ = ["DetectionResult", "YOLO26Detector"]


@dataclass
class DetectionResult:
    """Detection results for a batch of images."""

    detections: list[Detection]
    latency_ms: float
    n_images: int

    @property
    def ms_per_image(self) -> float:
        return self.latency_ms / max(self.n_images, 1)


class YOLO26Detector:
    """Production YOLO26 nano detector wrapper."""

    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imgsz: int = 640,
        default_conf: float = 0.25,
        iou: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.default_conf = default_conf
        self.iou = iou

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        imgsz: int = 640,
        default_conf: float = 0.25,
    ) -> "YOLO26Detector":
        """Load from a trained .pt or exported .onnx checkpoint."""
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights not found: {weights_path}")

        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        if device != "cpu" and torch.cuda.is_available():
            model.to(device)
        return cls(model, device=device, imgsz=imgsz, default_conf=default_conf)

    def params(self) -> int:
        """Total trainable parameters in the model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
            return sum(p.numel() for p in self.model.model.parameters())
        return 0

    @torch.no_grad()
    def predict(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        conf: float | None = None,
    ) -> DetectionResult:
        """Run object detection on a list of images."""
        t0 = time.perf_counter()
        c = conf if conf is not None else self.default_conf
        raw_inputs = [str(x) if isinstance(x, Path) else x for x in images]

        results = self.model.predict(
            raw_inputs,
            imgsz=self.imgsz,
            conf=c,
            iou=self.iou,
            max_det=50,
            device=self.device,
            verbose=False,
            stream=False,
        )

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                labels = boxes.cls.cpu().numpy().astype(int)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=int)

            detections.append(
                Detection(
                    boxes=xyxy,
                    scores=scores,
                    labels=labels,
                    gt_boxes=np.zeros((0, 4), dtype=np.float32),
                    gt_labels=np.zeros((0,), dtype=int),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            detections=detections,
            latency_ms=elapsed_ms,
            n_images=len(images),
        )

    def export_onnx(self, out_path: str | Path | None = None) -> Path:
        """Export the underlying model to ONNX for production edge runtime."""
        path = self.model.export(format="onnx", imgsz=self.imgsz, dynamic=False)
        return Path(path)

