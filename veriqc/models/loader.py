"""Construct the ONNX runtime models declared by a profile."""

from __future__ import annotations

from veriqc.config import Profile
from veriqc.models.onnx_runtime import ONNXDetector, ONNXPatchCore, ONNXSegmenter

__all__ = ["load_models"]


def load_models(profile: Profile, num_threads: int = 0):
    """Return the single vision channel configured for this profile."""

    runtime_paths = profile.runtime_artifact_paths()
    if profile.known is not None:
        spec = profile.known
        if spec.kind == "detector":
            detector = ONNXDetector(
                runtime_paths["known_runtime"],
                spec.classes,
                input_size=int(spec.input_size or 640),
                conf_threshold=spec.conf_threshold,
                iou_threshold=spec.iou_threshold,
                num_threads=num_threads,
            )
            return detector, None, None
        if spec.kind == "segmenter":
            width, height = (
                tuple(spec.input_size)
                if isinstance(spec.input_size, tuple)
                else (
                    int(spec.input_size or 384),
                    int(spec.input_size or 1024),
                )
            )
            segmenter = ONNXSegmenter(
                runtime_paths["known_runtime"],
                input_width=width,
                input_height=height,
                num_threads=num_threads,
            )
            return None, segmenter, None
        raise ValueError(f"unsupported known-channel type: {spec.kind!r}")

    if profile.anomaly is not None:
        spec = profile.anomaly
        anomaly = ONNXPatchCore(
            runtime_paths["anomaly_backbone"],
            runtime_paths["anomaly_banks"],
            input_size=spec.input_size,
            image_layer=spec.image_layer,
            pixel_layer=spec.pixel_layer,
            num_threads=num_threads,
        )
        return None, None, anomaly

    raise ValueError(f"profile {profile.profile_id!r} declares no vision channel")
