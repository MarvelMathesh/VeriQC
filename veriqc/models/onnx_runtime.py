"""ONNX Runtime execution for the production CPU inspection path.

The PyTorch wrappers remain the reference used for training, export and parity
checks.  This module implements the same preprocessing/postprocessing contracts
without importing torch, so an ONNX-only station does not pay the training
stack's import and memory cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from veriqc.models.types import AnomalyResult, Detection

__all__ = [
    "ONNXDetector",
    "ONNXSegmenter",
    "ONNXPatchCore",
    "ONNXDetectorConfig",
    "ONNXSegmenterConfig",
    "ONNXPatchCoreConfig",
]

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass(frozen=True)
class ONNXDetectorConfig:
    input_size: int = 640
    conf_threshold: float = 0.01
    iou_threshold: float = 0.5
    max_detections: int = 100
    num_threads: int = 0


@dataclass(frozen=True)
class ONNXSegmenterConfig:
    input_width: int = 384
    input_height: int = 1024
    num_threads: int = 0


@dataclass(frozen=True)
class ONNXPatchCoreConfig:
    input_size: int = 448
    backbone: str = "resnet18"
    image_layer: str = "layer2"
    pixel_layer: str = "layer3"
    bank_size: int = 4000
    aggr_kernel: int = 3
    num_threads: int = 0


def _session(path: str | Path, num_threads: int = 0):
    import onnxruntime as ort

    if not Path(path).exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        options.intra_op_num_threads = int(num_threads)
        options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _check_input_shape(session, expected: tuple[int, ...], label: str) -> None:
    """Reject a static ONNX input geometry that disagrees with the profile."""

    actual = session.get_inputs()[0].shape
    if len(actual) != len(expected):
        raise ValueError(f"{label}: input rank {actual} does not match {expected}")
    for axis, (value, want) in enumerate(zip(actual, expected)):
        if isinstance(value, int) and value > 0 and value != want:
            raise ValueError(
                f"{label}: input axis {axis} is {value}, expected {want} ({actual})"
            )


def _letterbox_bgr(image: Image.Image, size: int) -> tuple[np.ndarray, float, int, int, tuple[int, int]]:
    """Ultralytics-compatible letterbox: scale up, centre, pad with grey 114."""

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    scale = min(size / height, size / width)
    new_w = max(round(width * scale), 1)
    new_h = max(round(height * scale), 1)
    if (new_w, new_h) != (width, height):
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = size - new_w
    pad_h = size - new_h
    left = round(pad_w / 2 - 0.1)
    right = round(pad_w / 2 + 0.1)
    top = round(pad_h / 2 - 0.1)
    bottom = round(pad_h / 2 + 0.1)
    if pad_w or pad_h:
        bgr = cv2.copyMakeBorder(
            bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    return tensor, scale, left, top, (width, height)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float, max_det: int):
    """Class-aware greedy NMS on xyxy boxes."""

    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            a = boxes[current]
            b = boxes[rest]
            inter_x1 = np.maximum(a[0], b[:, 0])
            inter_y1 = np.maximum(a[1], b[:, 1])
            inter_x2 = np.minimum(a[2], b[:, 2])
            inter_y2 = np.minimum(a[3], b[:, 3])
            inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
            union = area_a + area_b - inter
            iou = inter / np.maximum(union, 1e-9)
            order = rest[iou <= iou_threshold]
    keep = keep[:max_det]
    return np.asarray(keep, dtype=np.int64)


class ONNXDetector:
    def __init__(
        self,
        model_path: str | Path,
        class_names: Sequence[str],
        *,
        input_size: int = 640,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        num_threads: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = tuple(str(name) for name in class_names)
        self.config = ONNXDetectorConfig(
            input_size=int(input_size),
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            max_detections=int(max_detections),
            num_threads=int(num_threads),
        )
        self.session = _session(self.model_path, num_threads)
        _check_input_shape(
            self.session,
            (1, 3, self.config.input_size, self.config.input_size),
            "detector",
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, image: Image.Image):
        started = time.perf_counter()
        tensor, scale, pad_x, pad_y, original_size = _letterbox_bgr(image, self.config.input_size)
        output = self.session.run(self.output_names, {self.input_name: tensor})[0]
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.shape[0] == 4 + len(self.class_names) and predictions.shape[1] != 4 + len(self.class_names):
            predictions = predictions.T
        if predictions.shape[1] < 4 + len(self.class_names):
            raise ValueError(f"unexpected detector output shape: {predictions.shape}")

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4 : 4 + len(self.class_names)]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        valid = scores >= self.config.conf_threshold
        if not np.any(valid):
            return [], (time.perf_counter() - started) * 1000.0

        boxes_xywh = boxes_xywh[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        keep = _nms(
            boxes,
            scores,
            class_ids,
            self.config.iou_threshold,
            self.config.max_detections,
        )
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        width, height = original_size
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    confidence=float(score),
                    bbox=tuple(float(v) for v in box),
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections, (time.perf_counter() - started) * 1000.0


class ONNXSegmenter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_width: int = 384,
        input_height: int = 1024,
        num_threads: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = ONNXSegmenterConfig(
            input_width=int(input_width),
            input_height=int(input_height),
            num_threads=int(num_threads),
        )
        self.session = _session(self.model_path, num_threads)
        _check_input_shape(
            self.session,
            (1, 3, self.config.input_height, self.config.input_width),
            "segmenter",
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict_proba(self, image: Image.Image) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        resized = image.convert("RGB").resize(
            (self.config.input_width, self.config.input_height), Image.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = array.transpose(2, 0, 1)[None]
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        logits = self.session.run(self.output_names, {self.input_name: tensor.astype(np.float32)})[0]
        probability = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)[0, 0]))
        heatmap = np.asarray(
            Image.fromarray(probability.astype(np.float32), mode="F").resize(
                image.size, Image.BILINEAR
            ),
            dtype=np.float32,
        )
        return heatmap, (time.perf_counter() - started) * 1000.0


class ONNXPatchCore:
    def __init__(
        self,
        backbone_path: str | Path,
        banks_path: str | Path,
        *,
        input_size: int = 448,
        image_layer: str = "layer2",
        pixel_layer: str = "layer3",
        num_threads: int = 0,
        aggr_kernel: int = 3,
    ) -> None:
        self.backbone_path = Path(backbone_path)
        self.banks_path = Path(banks_path)
        banks = np.load(self.banks_path, allow_pickle=False)
        self.image_bank = np.ascontiguousarray(banks["image_bank"], dtype=np.float32)
        self.pixel_bank = np.ascontiguousarray(banks["pixel_bank"], dtype=np.float32)
        if self.image_bank.ndim != 2 or self.pixel_bank.ndim != 2:
            raise ValueError("anomaly banks must be two-dimensional patch matrices")
        if self.image_bank.shape[0] != self.pixel_bank.shape[0]:
            raise ValueError("image and pixel banks must contain the same number of patches")
        self._image_bank_t = np.ascontiguousarray(self.image_bank.T)
        self._pixel_bank_t = np.ascontiguousarray(self.pixel_bank.T)

        bank_image_layer = (
            str(banks["image_layer"]) if "image_layer" in banks else str(image_layer)
        )
        bank_pixel_layer = (
            str(banks["pixel_layer"]) if "pixel_layer" in banks else str(pixel_layer)
        )
        bank_input_size = (
            int(banks["input_size"]) if "input_size" in banks else int(input_size)
        )
        if bank_image_layer != str(image_layer) or bank_pixel_layer != str(pixel_layer):
            raise ValueError(
                "anomaly bank layers do not match the profile: "
                f"bank={bank_image_layer}/{bank_pixel_layer}, "
                f"profile={image_layer}/{pixel_layer}"
            )
        if bank_input_size != int(input_size):
            raise ValueError(
                f"anomaly bank input size {bank_input_size} does not match profile {input_size}"
            )

        self.config = ONNXPatchCoreConfig(
            input_size=int(input_size),
            image_layer=str(image_layer),
            pixel_layer=str(pixel_layer),
            bank_size=int(self.image_bank.shape[0]),
            aggr_kernel=int(aggr_kernel),
            num_threads=int(num_threads),
        )
        self.session = _session(self.backbone_path, num_threads)
        _check_input_shape(
            self.session,
            (1, 3, self.config.input_size, self.config.input_size),
            "anomaly backbone",
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        missing = {self.config.image_layer, self.config.pixel_layer} - set(
            self.output_names
        )
        if missing:
            raise ValueError(
                f"anomaly backbone is missing output layers {sorted(missing)}; "
                f"available outputs are {self.output_names}"
            )

    @property
    def bank_size(self) -> int:
        return int(self.image_bank.shape[0])

    def _preprocess(self, image: Image.Image):
        size = self.config.input_size
        width, height = image.size
        scale = size / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        ox = (size - new_w) // 2
        oy = (size - new_h) // 2
        resized = image.convert("RGB").resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(resized, (ox, oy))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = array.transpose(2, 0, 1)[None]
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        return tensor.astype(np.float32), (ox, oy, new_w, new_h)

    def _stage_features(self, features: np.ndarray, layer: str) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim != 4:
            raise ValueError(f"unexpected {layer} feature shape: {value.shape}")
        kernel = self.config.aggr_kernel
        pad = kernel // 2
        padded = np.pad(value, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        pooled = np.zeros_like(value)
        for dy in range(kernel):
            for dx in range(kernel):
                pooled += padded[:, :, dy : dy + value.shape[2], dx : dx + value.shape[3]]
        pooled /= float(kernel * kernel)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-6)
        return pooled / norm

    @staticmethod
    def _min_distance(
        patches: np.ndarray,
        bank: np.ndarray,
        bank_t: np.ndarray | None = None,
        chunk: int = 2048,
    ) -> np.ndarray:
        if patches.ndim != 4:
            raise ValueError(f"expected NCHW patch features, got {patches.shape}")
        flat = patches.transpose(0, 2, 3, 1).reshape(-1, patches.shape[1]).astype(np.float32)
        bank = bank.astype(np.float32)
        if bank_t is None:
            bank_t = np.ascontiguousarray(bank.T)
        bank_norm = np.einsum("ij,ij->i", bank, bank)[None, :]
        best = np.empty(flat.shape[0], dtype=np.float32)
        for start in range(0, flat.shape[0], chunk):
            block = flat[start : start + chunk]
            block_norm = np.einsum("ij,ij->i", block, block)[:, None]
            distance_sq = block_norm + bank_norm - 2.0 * np.dot(block, bank_t)
            np.maximum(distance_sq, 0.0, out=distance_sq)
            best[start : start + chunk] = np.sqrt(distance_sq).min(axis=1)
        return best

    def score(self, image: Image.Image) -> AnomalyResult:
        started = time.perf_counter()
        tensor, content = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        features = {name: value for name, value in zip(self.output_names, outputs)}
        image_features = self._stage_features(features[self.config.image_layer], self.config.image_layer)
        pixel_features = self._stage_features(features[self.config.pixel_layer], self.config.pixel_layer)
        image_distance = self._min_distance(
            image_features, self.image_bank, self._image_bank_t
        )
        pixel_distance = self._min_distance(
            pixel_features, self.pixel_bank, self._pixel_bank_t
        )
        image_score = float(image_distance.max())
        score_map = pixel_distance.reshape(
            1, 1, pixel_features.shape[2], pixel_features.shape[3]
        )
        upsampled = cv2.resize(
            score_map[0, 0],
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        ox, oy, new_w, new_h = content
        crop = upsampled[oy : oy + new_h, ox : ox + new_w]
        heatmap = (
            np.zeros((image.height, image.width), dtype=np.float32)
            if crop.size == 0
            else cv2.resize(crop, image.size, interpolation=cv2.INTER_LINEAR)
        )
        return AnomalyResult(
            image_score=image_score,
            heatmap=heatmap,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
