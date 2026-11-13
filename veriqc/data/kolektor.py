"""KolektorSDD dataset loader: electrical commutator surface crack inspection.

Dataset Characteristics:
- 49 distinct parts (kos01 to kos49), 399 total images (500 x ~1240 px).
- 52 defective images with micro-cracks, 347 defect-free normal images.
- Crack area: median 3,367 pixels (0.53% of frame area).

Splitting Policy:
Each part represents 8 scans of the same physical commutator. Splits are strictly
performed by physical part (never by image) to prevent data leakage between training,
calibration, and test sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from veriqc.data.manifest import DATA_ROOT

__all__ = [
    "KolektorSample",
    "Kolektor",
    "load_kolektor",
    "list_parts",
    "bbox_from_mask",
]

KOLEKTOR_ROOT = DATA_ROOT / "kolektor"


def bbox_from_mask(mask: "np.ndarray") -> tuple[float, float, float, float] | None:
    """Bounding box (x1, y1, x2, y2) of a binary mask, or None if empty."""
    import numpy as np

    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


@dataclass
class KolektorSample:
    """One image with its crack mask (if any) and its crack box (if any)."""

    part: str
    name: str
    image: Path
    mask: Path
    boxes_mask: Path | None = None
    defective: bool = False
    area_px: int = 0
    box: tuple[float, float, float, float] | None = None
    width: int = 0
    height: int = 0

    @property
    def aspect(self) -> float:
        return self.height / max(1, self.width)


@dataclass
class Kolektor:
    """The whole dataset, indexed by part."""

    root: Path
    samples: list[KolektorSample] = field(default_factory=list)

    # -- views -------------------------------------------------------------
    @property
    def normals(self) -> list[KolektorSample]:
        return [s for s in self.samples if not s.defective]

    @property
    def anomalies(self) -> list[KolektorSample]:
        return [s for s in self.samples if s.defective]

    @property
    def parts(self) -> list[str]:
        return sorted({s.part for s in self.samples})

    def by_part(self) -> dict[str, list[KolektorSample]]:
        out: dict[str, list[KolektorSample]] = {}
        for s in self.samples:
            out.setdefault(s.part, []).append(s)
        return out

    def split_by_part(
        self, fit_fraction: float = 0.5, seed: int = 0
    ) -> tuple[list[KolektorSample], list[KolektorSample], list[KolektorSample]]:
        """Split into (fit, calibrate, test) by part.

        The split is over *parts*, because the eight images of a part are the
        same physical object scanned eight times.  A split over images would
        put a near-duplicate of every test image into the memory bank.
        """
        import random

        parts = self.parts
        rng = random.Random(seed)
        rng.shuffle(parts)
        n_fit = int(round(len(parts) * fit_fraction))
        rest = parts[n_fit:]
        n_calib = len(rest) // 2
        fit_p = set(parts[:n_fit])
        calib_p = set(rest[:n_calib])
        test_p = set(rest[n_calib:])
        fit = [s for s in self.samples if s.part in fit_p]
        calib = [s for s in self.samples if s.part in calib_p]
        test = [s for s in self.samples if s.part in test_p]
        return fit, calib, test

    def summary(self) -> dict:
        return {
            "parts": len(self.parts),
            "images": len(self.samples),
            "normals": len(self.normals),
            "defective": len(self.anomalies),
            "defect_rate": round(len(self.anomalies) / max(1, len(self.samples)), 4),
            "median_crack_area_px": (
                sorted(s.area_px for s in self.anomalies)[len(self.anomalies) // 2]
                if self.anomalies
                else 0
            ),
        }


def list_parts(root: Path = KOLEKTOR_ROOT) -> list[str]:
    return sorted(p.name for p in root.glob("kos*") if p.is_dir() and (p / "Part0.jpg").exists())


def load_kolektor(root: Path = KOLEKTOR_ROOT) -> Kolektor:
    """Load every image, its crack mask and its crack box."""
    import numpy as np
    from PIL import Image

    samples: list[KolektorSample] = []
    for part in list_parts(root):
        pdir = root / part
        boxes_dir = root / "boxes" / part
        for img_path in sorted(pdir.glob("Part*.jpg")):
            stem = img_path.stem
            mask_path = pdir / f"{stem}_label.bmp"
            if not mask_path.exists():
                continue
            box_mask_path = boxes_dir / f"{stem}_label.bmp"
            if not box_mask_path.exists():
                box_mask_path = None

            mask = np.array(Image.open(mask_path).convert("L"))
            area = int((mask > 0).sum())
            box = bbox_from_mask(mask)
            height, width = mask.shape[:2]
            samples.append(
                KolektorSample(
                    part=part,
                    name=stem,
                    image=img_path,
                    mask=mask_path,
                    boxes_mask=box_mask_path,
                    defective=area > 0,
                    area_px=area,
                    box=box,
                    width=int(width),
                    height=int(height),
                )
            )
    return Kolektor(root=root, samples=samples)


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(load_kolektor().summary(), indent=2))
