"""VeriQC Vision Module.

Exports the winning edge vision architecture:
- PatchEmbedder (EfficientNet-B0 default backbone)
- AnomalyGate (Greedy Coreset-kNN memory bank)
- YOLO26Detector (Ultralytics YOLO26 nano)
- VisionPipeline (Integrated end-to-end vision inspection)
"""

from veriqc.vision.backbone import PatchEmbedder
from veriqc.vision.detector import DetectionResult, YOLO26Detector
from veriqc.vision.gate import AnomalyGate, GateConfig, GateScore
from veriqc.vision.pipeline import PipelineConfig, VisionPipeline

__all__ = [
    "PatchEmbedder",
    "YOLO26Detector",
    "DetectionResult",
    "AnomalyGate",
    "GateConfig",
    "GateScore",
    "VisionPipeline",
    "PipelineConfig",
]

