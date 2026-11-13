"""Build the combined detector dataset: NEU-DET + KolektorSDD, one taxonomy.

Rationale
---------
The station runs **one** detector at the edge. Training a separate model per
dataset would mean a per-product model zoo, which the edge deployment
architecture explicitly avoids. The defect datasets are trained as one model on one
taxonomy, and the per-dataset breakdown is reported separately so results can
be analyzed per domain.

Taxonomy (7 classes)::

    0 crazing        NEU-DET
    1 inclusion      NEU-DET
    2 patches        NEU-DET
    3 pitted_surface NEU-DET
    4 rolled-in_scale NEU-DET
    5 scratches      NEU-DET
    6 crack          KolektorSDD

``crack`` stays a separate class rather than being folded into ``crazing``.
Both are crack-like, but they are different defects on different hardware: a
commutator surface crack and a hot-rolled steel network of fine cracks. Folding
them together would corrupt the NEU-DET confusion matrix during per-class analysis.

Split policy
------------
* NEU-DET keeps its deterministic 8:1:1 split and its official 360-image
  ``validation`` folder stays out of training, exactly as in
  ``veriqc/data/convert_neu.py``.
* KolektorSDD is split **by part**.  The eight images of one part are eight
  scans of the same physical commutator, so a split over images would put a
  near-duplicate of every test image into the training set and inflate the
  result.  A part-wise split is the only honest one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from veriqc.data.kolektor import load_kolektor
from veriqc.data.manifest import DATA_ROOT
from veriqc.data.neu import NEU_CLASSES, load_neu

__all__ = ["CLASS_NAMES", "convert_combined", "write_dataset_yaml"]

YOLO_ROOT = DATA_ROOT / "combined_yolo"

#: Unified taxonomy.  NEU's six classes keep their index; Kolektor's crack is 6.
CLASS_NAMES: list[str] = [*NEU_CLASSES, "crack"]
CRACK_CLASS_ID = len(NEU_CLASSES)


def convert_neu_to_yolo(seed: int = 0, root: Path | None = None, link: bool = True) -> None:
    """Convert NEU-DET annotations to YOLO format."""
    target_root = root or (DATA_ROOT / "neu_yolo")
    neu = load_neu(seed=seed)
    for split_name, samples in [("train", neu.train), ("val", neu.val), ("test", neu.test)]:
        img_dir = target_root / "images" / split_name
        lbl_dir = target_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            dst_img = img_dir / f"{s.name}.jpg"
            if dst_img.exists():
                dst_img.unlink()
            if link:
                dst_img.symlink_to(s.path.resolve())
            else:
                shutil.copy2(s.path, dst_img)

            lines = []
            for box, label in zip(s.boxes, s.labels):
                x1, y1, x2, y2 = box
                xc = ((x1 + x2) / 2.0) / s.width
                yc = ((y1 + y2) / 2.0) / s.height
                w = (x2 - x1) / s.width
                h = (y2 - y1) / s.height
                lines.append(f"{label} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{s.name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def _split_kolektor(seed: int = 0) -> dict[str, list]:
    """Part-wise split into train/val/test with every split holding cracks.

    Greedy: parts are shuffled, then assigned to the split that is furthest
    below its quota, subject to the constraint that a split only takes a
    defective part while it still has no defect.
    """
    import random

    ds = load_kolektor()
    by_part = ds.by_part()

    quotas = {"train": 0.70, "val": 0.15, "test": 0.15}
    parts = sorted(by_part)
    rng = random.Random(seed)
    rng.shuffle(parts)

    # seed each split with a defective part so no split is crack-free
    defective_parts = [p for p in parts if any(s.defective for s in by_part[p])]
    normal_parts = [p for p in parts if not any(s.defective for s in by_part[p])]
    rng.shuffle(defective_parts)
    rng.shuffle(normal_parts)

    assign: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    have_defect = {"train": False, "val": False, "test": False}
    for split in ("train", "val", "test"):
        if defective_parts:
            assign[split].append(defective_parts.pop())
            have_defect[split] = True

    total = len(parts)
    quotas = {"train": 0.70, "val": 0.15, "test": 0.15}
    remaining = normal_parts + defective_parts
    rng.shuffle(remaining)
    for p in remaining:
        n_def = sum(s.defective for s in by_part[p])
        # a split that still has no defect may take a defective part
        need_defect = [s for s in assign if not have_defect[s]]
        pool = need_defect if (n_def and need_defect) else list(assign)
        # pick the split that is furthest below its quota
        best = min(pool, key=lambda s: len(assign[s]) / (quotas[s] * total))
        assign[best].append(p)
        if n_def:
            have_defect[best] = True

    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for split, plist in assign.items():
        for p in plist:
            out[split].extend(by_part[p])
    return out


def convert_combined(seed: int = 0, root: Path = YOLO_ROOT, link: bool = True) -> Path:
    """Write the combined YOLO dataset and return the path of ``dataset.yaml``."""
    root.mkdir(parents=True, exist_ok=True)

    # --- NEU-DET: reuse the existing converter, then rewrite the class ids --
    neu_root = root / "neu"
    convert_neu_to_yolo(seed=seed, root=neu_root, link=link)
    for split in ("train", "val", "test"):
        img_dir = neu_root / "images" / split
        lbl_dir = neu_root / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            lines = [
                f"{int(parts[0])} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                for parts in (ln.split() for ln in lbl.read_text().splitlines() if ln.strip())
            ]
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

    # --- KolektorSDD: one class, boxes from the boxes distribution ---------
    ds = load_kolektor()
    splits = _split_kolektor(seed=seed)
    kol_stats = {}
    for split, samples in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_def = 0
        for s in samples:
            dst = img_dir / f"kol_{s.part}_{s.name}.jpg"
            if not dst.exists():
                try:
                    dst.symlink_to(s.image.resolve())
                except OSError:
                    shutil.copy2(s.image, dst)
            if s.defective and s.box is not None:
                x1, y1, x2, y2 = s.box
                w, h = s.width, s.height
                lines = [
                    f"{CRACK_CLASS_ID} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                    f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                ]
                n_def += 1
            else:
                lines = []
            (lbl_dir / f"kol_{s.part}_{s.name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        kol_stats[split] = {"images": len(samples), "with_crack": n_def}

    # --- merge -----------------------------------------------------------
    import os

    for split in ("train", "val", "test"):
        for src in (neu_root / "images" / split, root / "images" / split):
            if not src.exists():
                continue
            for f in src.iterdir():
                dst = root / "images" / split / f.name
                if dst.exists():
                    continue
                # skip the intermediate folder we are copying out of
                if src.resolve() == dst.resolve():
                    continue
                try:
                    os.symlink(f.resolve(), dst)
                except OSError:
                    shutil.copy2(f, dst)
        for src in (neu_root / "labels" / split, root / "labels" / split):
            if not src.exists():
                continue
            for f in src.iterdir():
                dst = root / "labels" / split / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)

    yaml_path = write_dataset_yaml(root, seed=seed)
    return yaml_path


def write_dataset_yaml(root: Path = YOLO_ROOT, seed: int = 0) -> Path:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASS_NAMES))
    text = f"""# Combined detector dataset: NEU-DET (6 classes) + KolektorSDD (crack).
# NEU-DET keeps its deterministic 8:1:1 split (seed={seed}); its official
# 360-image 'validation' folder is held out and evaluated separately.
# KolektorSDD is split by PART (eight scans of one commutator are never
# split across train and test).
path: {root}
train: images/train
val: images/val
test: images/test

names:
{names}
"""
    p = root / "dataset.yaml"
    p.write_text(text)
    return p


if __name__ == "__main__":
    import json

    p = convert_combined()
    splits = _split_kolektor()
    print(f"wrote {p}")
    print(
        json.dumps(
            {
                "kolektor_splits": {
                    k: {"images": len(v), "with_crack": sum(s.defective for s in v)}
                    for k, v in splits.items()
                }
            },
            indent=2,
        )
    )
