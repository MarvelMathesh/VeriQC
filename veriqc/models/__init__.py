"""Vision model implementations.

The production runtime uses :mod:`veriqc.models.onnx_runtime`.  The
``torch_*`` modules are the training, calibration and parity reference and are
imported explicitly by the commands that need them.
"""

from __future__ import annotations
