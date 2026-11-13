"""MVTec AD 2 adapter -- the lighting-shift benchmark.

Layout on disk::

    data/mvtec_ad2/mvtec_ad_2/<scenario>/train/good/*.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/validation/good/*.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/test_public/good/<id>_<condition>.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/test_public/bad/<id>_<condition>.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/test_public/ground_truth/bad/<id>_<condition>.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/test_private/<id>_regular.png
    data/mvtec_ad2/mvtec_ad_2/<scenario>/test_private_mixed/<id>_mixed.png

The file name encodes the imaging condition, which is the whole point of this
dataset: ``regular``, ``overexposed``, ``underexposed``, ``shift_1..4``.  The
same physical defect is captured under several conditions, so a detector's
failure under lighting change is directly measurable rather than inferred.

``test_public`` is the *labelled* benchmark: for one defective scene the
dataset ships the same defect under six lighting conditions, each with a
pixel-precise mask.  That is the controlled experiment E3 uses.

``test_private`` and ``test_private_mixed`` are the same 321 scenes captured
under the regular and under a varied lighting regime.  The images are in the
public tarball, but the ground truth is not: MVTec keeps it on an evaluation
server, and the id counters in ``test_public/good`` and ``test_public/bad``
are independent, so a private image cannot be labelled from the public set.
They are therefore used for *unlabelled* drift measurement only -- how far a
score moves when nothing but the lighting changes -- never for accuracy
claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from veriqc.data.manifest import DATA_ROOT

__all__ = ["AD2Scenario", "AD2", "parse_condition", "list_scenarios"]

AD2_ROOT = DATA_ROOT / "mvtec_ad2"
_COND_RE = re.compile(r"^\d+_(?P<cond>.+)\.png$")


def parse_condition(filename: str) -> str:
    """Extract the imaging condition from an AD2 file name.

    ``000_overexposed.png`` -> ``overexposed``;  ``012_regular.png`` -> ``regular``.
    Unknown patterns fall back to ``unknown``.
    """
    m = _COND_RE.match(Path(filename).name)
    return m.group("cond") if m else "unknown"


@dataclass
class AD2Scenario:
    """One AD2 scenario with normals, calibration normals and a shift-stressed test set."""

    name: str
    root: Path
    train: list[Path] = field(default_factory=list)
    validation: list[Path] = field(default_factory=list)
    test_good: list[Path] = field(default_factory=list)
    test_bad: list[Path] = field(default_factory=list)
    masks: dict[Path, Path] = field(default_factory=dict)
    private: list[Path] = field(default_factory=list)
    private_mixed: list[Path] = field(default_factory=list)

    # populated on demand
    def condition_of(self, path: Path) -> str:
        return parse_condition(path.name)

    def normals(self) -> list[Path]:
        return self.train + self.validation

    def summary(self) -> dict:
        conds: dict[str, int] = {}
        for p in self.test_good + self.test_bad:
            c = self.condition_of(p)
            conds[c] = conds.get(c, 0) + 1
        return {
            "scenario": self.name,
            "train": len(self.train),
            "validation": len(self.validation),
            "normals_total": len(self.normals()),
            "test_good": len(self.test_good),
            "test_bad": len(self.test_bad),
            "conditions": conds,
            "private_regular": len(self.private),
            "private_mixed": len(self.private_mixed),
        }


def list_scenarios(root: Path = AD2_ROOT) -> list[str]:
    base = root / "mvtec_ad_2"
    if not base.exists():
        base = root
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "train").exists())


def _pngs(folder: Path) -> list[Path]:
    """PNG files under a folder, including one level of nesting (good/ bad/)."""
    out = list(folder.glob("*.png"))
    for sub in sorted(folder.iterdir()) if folder.exists() else []:
        if sub.is_dir():
            out.extend(sorted(sub.glob("*.png")))
    return sorted(out)


def load_scenario(name: str, root: Path = AD2_ROOT) -> AD2Scenario:
    base = root / "mvtec_ad_2"
    if not base.exists():
        base = root
    sroot = base / name
    train = _pngs(sroot / "train")
    val = _pngs(sroot / "validation")
    pub = sroot / "test_public"
    good = sorted((pub / "good").glob("*.png"))
    bad = sorted((pub / "bad").glob("*.png"))
    masks: dict[Path, Path] = {}
    gt_dir = pub / "ground_truth" / "bad"
    for img in bad:
        # the ground-truth files carry a "_mask" suffix: 006_shift_1.png ->
        # 006_shift_1_mask.png
        m = gt_dir / f"{img.stem}_mask.png"
        if m.exists():
            masks[img] = m
    # The private folders carry no ground truth (MVTec keeps it on an
    # evaluation server) and their id counters are independent of the public
    # ones, so they cannot be labelled.  They are loaded for unlabelled drift
    # measurement only.
    private = sorted((sroot / "test_private").glob("*.png"))
    private_mixed = sorted((sroot / "test_private_mixed").glob("*.png"))
    return AD2Scenario(
        name=name,
        root=sroot,
        train=train,
        validation=val,
        test_good=good,
        test_bad=bad,
        masks=masks,
        private=private,
        private_mixed=private_mixed,
    )


class AD2:
    def __init__(self, root: Path = AD2_ROOT):
        self.root = root
        self.scenarios = {s: load_scenario(s, root) for s in list_scenarios(root)}

    def __getitem__(self, name: str) -> AD2Scenario:
        return self.scenarios[name]

    def summary(self) -> list[dict]:
        return [s.summary() for s in self.scenarios.values()]
