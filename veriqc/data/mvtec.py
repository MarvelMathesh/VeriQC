"""MVTec AD adapter (15 categories, normals-only training).

Layout on disk::

    data/mvtec/<category>/train/good/*.png
    data/mvtec/<category>/test/good/*.png
    data/mvtec/<category>/test/<defect>/*.png
    data/mvtec/<category>/ground_truth/<defect>/*_mask.png

The train normals are split into three disjoint parts so that no image is
ever both a memory-bank exemplar and a calibration point:

    fit      -> fills the patch memory (the "normal" the bank represents)
    calib    -> the split-conformal calibration set (thresholds)
    holdout  -> normal test images used only for measuring the realised FPR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from veriqc.data.manifest import DATA_ROOT, split_list

__all__ = ["MVTecCategory", "MVTecAD", "list_categories"]

MVTEC_ROOT = DATA_ROOT / "mvtec"


@dataclass
class MVTecCategory:
    """One MVTec AD category with its three-way normal split."""

    name: str
    root: Path
    fit: list[Path] = field(default_factory=list)
    calib: list[Path] = field(default_factory=list)
    holdout: list[Path] = field(default_factory=list)
    test_good: list[Path] = field(default_factory=list)
    test_bad: list[Path] = field(default_factory=list)
    masks: dict[Path, Path] = field(default_factory=dict)
    defect_types: list[str] = field(default_factory=list)

    @property
    def n_fit(self) -> int:
        return len(self.fit)

    @property
    def n_calib(self) -> int:
        return len(self.calib)

    def summary(self) -> dict:
        return {
            "category": self.name,
            "fit": len(self.fit),
            "calib": len(self.calib),
            "holdout": len(self.holdout),
            "test_good": len(self.test_good),
            "test_bad": len(self.test_bad),
            "defect_types": list(self.defect_types),
        }


def list_categories(root: Path = MVTEC_ROOT) -> list[str]:
    base = root / "mvtec_anomaly_detection"
    if not base.exists():
        base = root
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "train").exists())


def load_category(
    name: str,
    root: Path = MVTEC_ROOT,
    seed: int = 0,
    fit_fraction: float = 0.5,
) -> MVTecCategory:
    """Load one category.

    ``fit_fraction`` is the share of the training normals that fills the
    memory bank; the rest is split evenly into the conformal calibration set
    and a normal holdout used only to measure the realised false alarm rate.
    The three parts are always disjoint.
    """
    base = root / "mvtec_anomaly_detection"
    if not base.exists():
        base = root
    croot = base / name
    train_good = sorted((croot / "train" / "good").glob("*.png"))
    rest = (1.0 - fit_fraction) / 2.0
    fit, calib, holdout = split_list(train_good, (fit_fraction, rest, rest), seed=seed)

    test_good = sorted((croot / "test" / "good").glob("*.png"))
    test_bad: list[Path] = []
    masks: dict[Path, Path] = {}
    defect_types = []
    for d in sorted((croot / "test").iterdir()):
        if not d.is_dir() or d.name == "good":
            continue
        defect_types.append(d.name)
        for img in sorted(d.glob("*.png")):
            test_bad.append(img)
            m = croot / "ground_truth" / d.name / f"{img.stem}_mask.png"
            if m.exists():
                masks[img] = m

    return MVTecCategory(
        name=name,
        root=croot,
        fit=fit,
        calib=calib,
        holdout=holdout,
        test_good=test_good,
        test_bad=test_bad,
        masks=masks,
        defect_types=defect_types,
    )


class MVTecAD:
    """Convenience wrapper over all categories."""

    def __init__(self, root: Path = MVTEC_ROOT, seed: int = 0, fit_fraction: float = 0.5):
        self.root = root
        self.seed = seed
        self.fit_fraction = fit_fraction
        self.categories = {
            c: load_category(c, root, seed, fit_fraction) for c in list_categories(root)
        }

    def __getitem__(self, name: str) -> MVTecCategory:
        return self.categories[name]

    def summary(self) -> list[dict]:
        return [c.summary() for c in self.categories.values()]
