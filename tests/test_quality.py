from __future__ import annotations

import numpy as np
from PIL import Image

from veriqc.quality import QualitySpec, assess_quality


def test_resolution_and_pixel_limits() -> None:
    tiny = Image.new("RGB", (10, 10), (120, 120, 120))
    report = assess_quality(tiny)
    assert "resolution" in report.flags
    report = assess_quality(Image.new("RGB", (20, 20)), QualitySpec(max_pixels=100))
    assert "pixel_count" in report.flags


def test_noise_frame_passes_default_gate() -> None:
    rng = np.random.default_rng(0)
    array = rng.integers(40, 210, size=(128, 128, 3), dtype=np.uint8)
    report = assess_quality(Image.fromarray(array, mode="RGB"))
    assert report.ok
    assert report.statistics["pixels"] == 128 * 128
