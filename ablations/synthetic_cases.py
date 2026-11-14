"""Synthetic Defect Record generator for advisory robustness and verifier stress-testing.

Generates diverse, reproducible DefectRecord scenarios to test verifier claim bounding:
- Nominal PASS / FAIL / REVIEW records across defect classes.
- Contradictory records (e.g., minor inclusion, high score with low confidence).
- Incomplete records (unseen defect with missing class or area).
- Optical quality failure flags.
- Multi-defect parts (co-occurring classes).
- Edge-case measurements (extreme area, near-boundary scores).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from veriqc.contract.record import DefectRecord

__all__ = ["SyntheticCase", "generate_cases", "CLASSES", "ZONES", "SEVERITIES"]

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
ZONES = ["edge", "center", "corner", "surface"]
SEVERITIES = ["minor", "major", "critical"]


@dataclass
class SyntheticCase:
    """A Defect Record plus the ground truth about what it contains."""

    record: DefectRecord
    kind: str  # clean_pass | clean_fail | clean_review | contradictory | missing_fields | multi | extreme
    expect_action_ids: bool = True
    expect_any_text: bool = True


def _record(
    part_id: str,
    verdict: str,
    defect_class: str | None,
    severity: str | None,
    zone: str | None,
    area_px: float | None,
    conf: float | None,
    score: float | None,
    flags: list[str] | None = None,
    reasons: list[str] | None = None,
) -> DefectRecord:
    area_mm2 = None
    if area_px is not None:
        area_mm2 = area_px * 0.25 * 0.25
    return DefectRecord(
        part_id=part_id,
        verdict=verdict,
        defect_class=defect_class,
        class_confidence=conf,
        severity=severity,
        zone=zone,
        mask_area_px=area_px,
        mask_area_mm2=area_mm2,
        anomaly_score=score,
        thresholds={"review": 1.0, "fail": 2.0},
        qc_flags=flags or [],
        qc_ok=not flags,
        reasons=reasons or [],
        model_versions={"gate": "g-1", "detector": "d-1"},
    )


def generate_cases(n: int = 200, seed: int = 0) -> list[SyntheticCase]:
    """Build ``n`` synthetic Defect Records covering the stress cases."""
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    kinds = [
        ("clean_pass", 0.15),
        ("clean_fail", 0.50),
        ("clean_review", 0.10),
        ("contradictory", 0.08),
        ("missing_fields", 0.07),
        ("multi", 0.05),
        ("extreme", 0.05),
    ]
    plan: list[str] = []
    for kind, frac in kinds:
        plan += [kind] * int(round(frac * n))
    while len(plan) < n:
        plan.append("clean_fail")
    rng.shuffle(plan)

    for i, kind in enumerate(plan[:n]):
        pid = f"P-{1000 + i}"
        cls = rng.choice(CLASSES)
        sev = rng.choice(SEVERITIES)
        zone = rng.choice(ZONES)
        area = round(rng.uniform(20, 900), 1)
        conf = round(rng.uniform(0.30, 0.99), 2)
        score = round(rng.uniform(2.05, 6.0), 3)

        if kind == "clean_pass":
            rec = _record(pid, "PASS", None, None, None, None, None, round(rng.uniform(0.05, 0.95), 3))
            cases.append(SyntheticCase(rec, kind, expect_action_ids=False))
        elif kind == "clean_fail":
            rec = _record(pid, "FAIL", cls, sev, zone, area, conf, score)
            cases.append(SyntheticCase(rec, kind))
        elif kind == "clean_review":
            rec = _record(pid, "REVIEW", cls, sev, zone, area, conf, round(rng.uniform(1.02, 1.98), 3))
            cases.append(SyntheticCase(rec, kind))
        elif kind == "contradictory":
            # a severe class graded as minor, or a minor class graded critical
            bad_sev = rng.choice(["minor", "critical"])
            rec = _record(pid, "FAIL", cls, bad_sev, zone, area, conf, score)
            cases.append(SyntheticCase(rec, kind))
        elif kind == "missing_fields":
            rec = _record(
                pid,
                rng.choice(["FAIL", "REVIEW"]),
                rng.choice([cls, None]),
                rng.choice([sev, None]),
                rng.choice([zone, None]),
                rng.choice([area, None]),
                rng.choice([conf, None]),
                score,
                flags=rng.choice([[], ["blur"], ["luminance"], ["blur", "saturation"]]),
            )
            cases.append(SyntheticCase(rec, kind))
        elif kind == "multi":
            rec = _record(pid, "FAIL", cls, sev, zone, area, conf, score)
            rec.extras["additional_defects"] = [rng.choice(CLASSES) for _ in range(rng.randint(1, 2))]
            cases.append(SyntheticCase(rec, kind))
        else:  # extreme
            rec = _record(
                pid,
                rng.choice(["FAIL", "REVIEW"]),
                cls,
                sev,
                zone,
                round(rng.uniform(900, 9000), 1),
                round(rng.choice([0.05, 0.99]), 2),
                round(rng.uniform(2.0, 20.0), 3),
            )
            cases.append(SyntheticCase(rec, kind))
    return cases
