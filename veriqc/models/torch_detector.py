"""Ultralytics detector wrapper for the known-defect channel."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Sequence

from PIL import Image

from veriqc.models.types import Detection

__all__ = ["Detector"]


class Detector:
    """A profile-owned YOLO detector, loaded from .pt or .onnx."""

    def __init__(
        self,
        model,
        class_names: Sequence[str],
        *,
        input_size: int = 640,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.class_names = tuple(str(name) for name in class_names)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        class_names: Sequence[str],
        *,
        input_size: int = 640,
        conf_threshold: float = 0.01,
        device: str = "cpu",
    ) -> "Detector":
        from ultralytics import YOLO, settings

        # The station performs all inference locally.  Disable the package's
        # optional analytics so the runtime has no incidental network traffic.
        settings.update({"sync": False})
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"detector weights not found: {path}")
        model = YOLO(str(path))
        resolved = tuple(str(name) for name in getattr(model, "names", {}).values())
        if resolved and tuple(str(name) for name in class_names) != resolved:
            raise ValueError(f"detector classes {resolved} do not match profile {tuple(class_names)}")
        return cls(
            model,
            class_names,
            input_size=input_size,
            conf_threshold=conf_threshold,
            device=device,
        )

    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        started = time.perf_counter()
        results = self.model.predict(
            [image],
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=100,
            device=self.device,
            verbose=False,
            stream=False,
        )
        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for box, score, class_id in zip(xyxy, scores, classes):
                    name = (
                        self.class_names[class_id]
                        if 0 <= class_id < len(self.class_names)
                        else f"class_{class_id}"
                    )
                    detections.append(
                        Detection(
                            class_id=int(class_id),
                            class_name=name,
                            confidence=float(score),
                            bbox=tuple(float(v) for v in box),
                        )
                    )
        elapsed = (time.perf_counter() - started) * 1000.0
        return detections, elapsed

    def export_onnx(self, out_path: str | Path | None = None) -> Path:
        exported = Path(
            self.model.export(
                format="onnx",
                imgsz=self.input_size,
                dynamic=False,
                half=False,
                simplify=True,
            )
        )
        if out_path is None:
            return exported
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, target)
        return target
