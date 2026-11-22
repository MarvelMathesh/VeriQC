"""Dataset preparation and split manifests.

The three sources are handled explicitly:

* NEU-DET: XML boxes converted to YOLO labels, official train/validation
  images, with a stratified 90/10 train/development split.
* KolektorSDD: the official JIM-2019 three-fold split by physical part;
  fine segmentation masks; scans of one part never cross folds.
* MVTec AD: official train/test, with a disjoint 80/20 fit/calibration split
  of ``train/good``.

No dataset is copied into the repository.  The manifests record source paths,
split membership and content hashes.
"""

from __future__ import annotations

import json
import pickle
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from veriqc import REPO_ROOT, sha256_file

__all__ = [
    "prepare_neu",
    "prepare_kolektor",
    "prepare_mvtec",
    "load_manifest",
    "split_by_part",
    "split_kolektor_parts",
    "split_kolektor_fixed",
]

PREPARED = REPO_ROOT / "data" / "prepared"
NEU_SOURCE = REPO_ROOT / "data" / "neu" / "NEU-DET"
KOLEKTOR_SOURCE = REPO_ROOT / "data" / "kolektor"
KOLEKTOR_SPLITS = REPO_ROOT / "data" / "kolektor_splits" / "split.pyb"
MVTEC_SOURCE = REPO_ROOT / "data" / "mvtec"

NEU_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


@dataclass(frozen=True)
class Item:
    image: str
    label: str | None
    split: str
    source: str
    part: str | None = None
    defect_class: str | None = None
    is_defect: bool = False
    sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _neu_boxes(xml_path: Path) -> list[tuple[int, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    boxes: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = obj.find("name")
        bbox = obj.find("bndbox")
        if name is None or bbox is None or name.text is None:
            continue
        try:
            class_id = NEU_CLASSES.index(name.text.strip())
        except ValueError:
            continue
        values = {}
        for key in ("xmin", "ymin", "xmax", "ymax"):
            node = bbox.find(key)
            if node is None or node.text is None:
                values = {}
                break
            values[key] = float(node.text)
        if not values:
            continue
        boxes.append(
            (
                class_id,
                values["xmin"],
                values["ymin"],
                values["xmax"],
                values["ymax"],
            )
        )
    return boxes


def prepare_neu(out_dir: Path = PREPARED / "neu", seed: int = 20261121) -> dict:
    """Create the YOLO-format NEU dataset and a JSON manifest."""

    if not NEU_SOURCE.is_dir():
        raise FileNotFoundError(f"NEU-DET source not found at {NEU_SOURCE}")

    items: list[Item] = []
    for official_split, target_split in (("train", "train"), ("validation", "test")):
        image_root = NEU_SOURCE / official_split / "images"
        for image in sorted(image_root.rglob("*.jpg")):
            xml = NEU_SOURCE / official_split / "annotations" / f"{image.stem}.xml"
            if not xml.exists():
                continue
            boxes = _neu_boxes(xml)
            if not boxes:
                continue
            width, height = _image_size(image)
            lines = []
            for class_id, x1, y1, x2, y2 in boxes:
                cx = (x1 + x2) / 2.0 / width
                cy = (y1 + y2) / 2.0 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not lines:
                continue
            items.append(
                Item(
                    image=str(image),
                    label="\n".join(lines),
                    split=target_split,
                    source="neu-det",
                    part=image.stem,
                    defect_class=image.parent.name,
                    is_defect=True,
                    sha256=sha256_file(image),
                )
            )

    rng = random.Random(seed)
    by_class: dict[str, list[Item]] = {}
    for item in items:
        if item.split == "train":
            by_class.setdefault(item.defect_class or "", []).append(item)
    train_items: list[Item] = []
    dev_items: list[Item] = []
    for key in sorted(by_class):
        group = by_class[key]
        rng.shuffle(group)
        n_dev = max(1, int(round(0.10 * len(group))))
        dev_items.extend(group[:n_dev])
        train_items.extend(group[n_dev:])
    test_items = [item for item in items if item.split == "test"]

    yolo_root = out_dir
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    for split, group in (("train", train_items), ("val", dev_items), ("test", test_items)):
        for index, item in enumerate(group):
            source = Path(item.image)
            image_target = yolo_root / "images" / split / source.name
            label_target = yolo_root / "labels" / split / f"{source.stem}.txt"
            _link_or_copy(source, image_target)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(item.label + "\n")

    dataset_yaml = yolo_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                *[f"  {i}: {name}" for i, name in enumerate(NEU_CLASSES)],
                "",
            ]
        )
    )

    manifest = {
        "schema": "veriqc.manifest/1",
        "source": "neu-det",
        "task": "detect",
        "classes": list(NEU_CLASSES),
        "seed": seed,
        "dataset_yaml": str(dataset_yaml),
        "counts": {
            "train": len(train_items),
            "val": len(dev_items),
            "test": len(test_items),
        },
        "items": {split: [item.to_dict() for item in group] for split, group in (
            ("train", train_items),
            ("val", dev_items),
            ("test", test_items),
        )},
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _kolektor_parts() -> list[str]:
    return sorted(
        path.name
        for path in KOLEKTOR_SOURCE.iterdir()
        if path.is_dir() and path.name.startswith("kos")
    )


def _kolektor_folds() -> list[list[str]]:
    if not KOLEKTOR_SPLITS.exists():
        raise FileNotFoundError(
            "the official Kolektor JIM-2019 split file is missing at "
            f"{KOLEKTOR_SPLITS}; the random fallback is not an official protocol"
        )
    with open(KOLEKTOR_SPLITS, "rb") as handle:
        _train, test, _all = pickle.load(handle, encoding="latin1")
    return [[str(part) for part in fold] for fold in test]


def prepare_kolektor(out_dir: Path = PREPARED / "kolektor") -> dict:
    """Build the official three-fold Kolektor manifest."""

    if not KOLEKTOR_SOURCE.is_dir():
        raise FileNotFoundError(f"KolektorSDD source not found at {KOLEKTOR_SOURCE}")
    folds = _kolektor_folds()
    all_folds = [set(fold) for fold in folds]
    overlaps = [
        (i, j, sorted(all_folds[i] & all_folds[j]))
        for i in range(len(folds))
        for j in range(i + 1, len(folds))
        if all_folds[i] & all_folds[j]
    ]
    if overlaps:
        raise RuntimeError(f"official Kolektor folds overlap: {overlaps}")

    fold_payload = []
    for fold_index, test_parts in enumerate(folds):
        train_parts = sorted(set().union(*[set(f) for j, f in enumerate(folds) if j != fold_index]))
        train_items: list[Item] = []
        test_items: list[Item] = []
        for part in sorted(set(train_parts) | set(test_parts)):
            part_dir = KOLEKTOR_SOURCE / part
            for image in sorted(part_dir.glob("*.jpg")):
                mask = image.with_name(image.stem + "_label.bmp")
                if not mask.exists():
                    continue
                is_defect = _mask_has_defect(mask)
                item = Item(
                    image=str(image),
                    label=str(mask),
                    split="train" if part in train_parts else "test",
                    source="kolektor-sdd",
                    part=part,
                    defect_class="crack" if is_defect else None,
                    is_defect=is_defect,
                    sha256=sha256_file(image),
                )
                (train_items if item.split == "train" else test_items).append(item)
        fold_payload.append(
            {
                "fold": fold_index,
                "train_parts": train_parts,
                "test_parts": sorted(test_parts),
                "train": [item.to_dict() for item in train_items],
                "test": [item.to_dict() for item in test_items],
                "counts": {
                    "train_images": len(train_items),
                    "train_defect_images": sum(i.is_defect for i in train_items),
                    "test_images": len(test_items),
                    "test_defect_images": sum(i.is_defect for i in test_items),
                },
            }
        )

    manifest = {
        "schema": "veriqc.manifest/1",
        "source": "kolektor-sdd",
        "task": "segment",
        "classes": ["crack"],
        "folds": fold_payload,
        "note": "Official JIM-2019 three-fold split by physical part.",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _mask_has_defect(path: Path) -> bool:
    import numpy as np
    from PIL import Image

    with Image.open(path) as mask:
        return bool(np.asarray(mask.convert("L")).any())


def prepare_mvtec(
    categories: Iterable[str] = (
        "carpet",
        "leather",
        "bottle",
        "capsule",
        "pill",
        "metal_nut",
        "screw",
        "transistor",
        "grid",
        "tile",
        "wood",
        "zipper",
        "cable",
        "hazelnut",
        "toothbrush",
    ),
    out_dir: Path = PREPARED / "mvtec",
    seed: int = 20261121,
) -> dict:
    """Build fit/calibration/test manifests for MVTec AD categories."""

    categories_payload = {}
    rng = random.Random(seed)
    for category in categories:
        root = MVTEC_SOURCE / category
        if not (root / "train" / "good").is_dir():
            continue
        good = sorted((root / "train" / "good").glob("*.png"))
        rng.shuffle(good)
        n_cal = max(5, int(round(0.20 * len(good))))
        cal = good[:n_cal]
        fit = good[n_cal:]
        items = []
        for split, group in (("fit", fit), ("cal", cal)):
            for image in group:
                items.append(
                    Item(
                        image=str(image),
                        label=None,
                        split=split,
                        source=f"mvtec:{category}",
                        part=image.stem,
                        is_defect=False,
                        sha256=sha256_file(image),
                    )
                )
        test_root = root / "test"
        for image in sorted(test_root.rglob("*.png")):
            if "ground_truth" in image.parts:
                continue
            rel = image.relative_to(test_root)
            defect_class = rel.parts[0]
            if defect_class == "good":
                mask = None
                is_defect = False
            else:
                mask_path = root / "ground_truth" / defect_class / f"{image.stem}_mask.png"
                mask = str(mask_path) if mask_path.exists() else None
                is_defect = True
            items.append(
                Item(
                    image=str(image),
                    label=mask,
                    split="test",
                    source=f"mvtec:{category}",
                    part=image.stem,
                    defect_class=None if not is_defect else defect_class,
                    is_defect=is_defect,
                    sha256=sha256_file(image),
                )
            )
        categories_payload[category] = {
            "root": str(root),
            "counts": {
                "fit": len(fit),
                "cal": len(cal),
                "test": sum(1 for i in items if i.split == "test"),
                "test_defect": sum(1 for i in items if i.split == "test" and i.is_defect),
            },
            "items": [item.to_dict() for item in items],
        }

    manifest = {
        "schema": "veriqc.manifest/1",
        "source": "mvtec-ad",
        "task": "anomaly",
        "seed": seed,
        "categories": categories_payload,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def split_by_part(items: list[dict], seed: int, val_fraction: float = 0.15) -> tuple[list[dict], list[dict]]:
    """Split scans by physical part so no part crosses the split."""
    parts = sorted({str(item.get("part")) for item in items})
    random.Random(seed).shuffle(parts)
    n_val = max(2, int(round(val_fraction * len(parts)))) if len(parts) > 2 else 0
    val_parts = set(parts[:n_val])
    train = [item for item in items if str(item.get("part")) not in val_parts]
    val = [item for item in items if str(item.get("part")) in val_parts]
    return train, val


def split_kolektor_fixed(
    items: list[dict],
    seed: int,
    *,
    cal_fraction: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Hold out calibration parts while keeping every other part in training.

    The training budget is fixed; no validation split is used for checkpoint
    selection.  This removes the shared calibration/early-stopping split
    without reducing the training population.
    """
    parts = sorted({str(item.get("part")) for item in items})
    random.Random(seed).shuffle(parts)
    n_cal = max(2, int(round(cal_fraction * len(parts)))) if len(parts) > 4 else 0
    cal_parts = set(parts[:n_cal])
    train = [item for item in items if str(item.get("part")) not in cal_parts]
    cal = [item for item in items if str(item.get("part")) in cal_parts]
    return train, [], cal


def split_kolektor_parts(
    items: list[dict],
    seed: int,
    *,
    val_fraction: float = 0.15,
    cal_fraction: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Three-way physical-part split for training, early stopping and calibration.

    The same part never appears in more than one role.  For the 33 training
    parts in a Kolektor fold this yields 23 train, 5 validation and 5
    calibration parts.
    """
    parts = sorted({str(item.get("part")) for item in items})
    random.Random(seed).shuffle(parts)
    n_val = max(2, int(round(val_fraction * len(parts)))) if len(parts) > 2 else 0
    n_cal = max(2, int(round(cal_fraction * len(parts)))) if len(parts) > 4 else 0
    if n_val + n_cal >= len(parts):
        raise ValueError("not enough parts for a three-way split")
    val_parts = set(parts[:n_val])
    cal_parts = set(parts[n_val : n_val + n_cal])
    train = [item for item in items if str(item.get("part")) not in val_parts | cal_parts]
    val = [item for item in items if str(item.get("part")) in val_parts]
    cal = [item for item in items if str(item.get("part")) in cal_parts]
    return train, val, cal
