"""NEU-DET adapter (hot-rolled steel surface defects, 6 classes).

The archive we hold contains ``train/`` (1440 images) and ``validation/``
(360 images) with VOC-style XML annotations.  The published NEU-DET papers
almost all use an 8:1:1 split, so we rebuild that split *from the 1440 train
images* (deterministically, seeded) and keep the official 360-image validation
folder as a second, never-trained-on held-out set.

Both sets are reported, because a detector that only looks good on one of
them is not good enough for a plant.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from veriqc.data.manifest import DATA_ROOT, split_list

__all__ = ["NEU_CLASSES", "NeuSample", "NEUDET", "load_neu"]

NEU_ROOT = DATA_ROOT / "neu" / "NEU-DET"

#: canonical class order (kept stable across every experiment and report)
NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {c: i for i, c in enumerate(NEU_CLASSES)}


@dataclass
class NeuSample:
    """One NEU-DET image with its boxes."""

    path: Path
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    split: str = "train"

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def classes(self) -> list[str]:
        return [NEU_CLASSES[i] for i in self.labels]


def _parse_xml(xml_path: Path) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text) if size is not None else 200
    h = int(size.find("height").text) if size is not None else 200
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        objs.append((int(x1), int(y1), int(x2), int(y2), CLASS_TO_ID.get(name, -1)))
    return w, h, objs


def _load_folder(folder: Path, split: str) -> list[NeuSample]:
    samples = []
    if not folder.exists():
        return samples
    for img in sorted(folder.rglob("*.jpg")):
        xml = folder.parent / "annotations" / f"{img.stem}.xml"
        if not xml.exists():
            xml = folder.parent / "annotations" / img.parent.name / f"{img.stem}.xml"
        if not xml.exists():
            continue
        w, h, objs = _parse_xml(xml)
        boxes, labels = [], []
        for x1, y1, x2, y2, cid in objs:
            if cid < 0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cid)
        samples.append(
            NeuSample(
                path=img,
                width=w,
                height=h,
                boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
                labels=np.array(labels, dtype=np.int64),
                split=split,
            )
        )
    return samples


class NEUDET:
    """NEU-DET with a deterministic 8:1:1 split and the official validation set."""

    def __init__(self, root: Path = NEU_ROOT, seed: int = 0):
        self.root = root
        self.seed = seed
        train = _load_folder(root / "train" / "images", "train_pool")
        val = _load_folder(root / "validation" / "images", "official_val")
        tr, va, te = split_list(train, (0.8, 0.1, 0.1), seed=seed)
        for s in tr:
            s.split = "train"
        for s in va:
            s.split = "val"
        for s in te:
            s.split = "test"
        self.train = tr
        self.val = va
        self.test = te
        self.official_val = val

    @property
    def classes(self) -> list[str]:
        return NEU_CLASSES

    def class_counts(self, split: list[NeuSample]) -> dict[str, int]:
        counts = {c: 0 for c in NEU_CLASSES}
        for s in split:
            for cid in s.labels:
                counts[NEU_CLASSES[cid]] += 1
        return counts

    def summary(self) -> dict:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "official_val": len(self.official_val),
            "train_class_counts": self.class_counts(self.train),
            "test_class_counts": self.class_counts(self.test),
            "official_val_class_counts": self.class_counts(self.official_val),
        }


def load_neu(seed: int = 0) -> NEUDET:
    return NEUDET(seed=seed)
