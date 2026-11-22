from __future__ import annotations

from pathlib import Path

import pytest

from veriqc.datasets import NEU_CLASSES, load_manifest

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = (
    REPO / "data" / "prepared" / "neu" / "manifest.json",
    REPO / "data" / "prepared" / "kolektor" / "manifest.json",
    REPO / "data" / "prepared" / "mvtec" / "manifest.json",
)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in MANIFESTS),
    reason="prepared dataset manifests are not present",
)


def test_neu_manifest_counts() -> None:
    manifest = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")
    assert manifest["classes"] == list(NEU_CLASSES)
    assert manifest["counts"]["train"] > 1000
    assert manifest["counts"]["val"] > 50
    assert manifest["counts"]["test"] == 360
    train_images = {item["image"] for item in manifest["items"]["train"]}
    test_images = {item["image"] for item in manifest["items"]["test"]}
    assert not train_images & test_images


def test_kolektor_folds_do_not_share_parts() -> None:
    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    assert len(manifest["folds"]) == 3
    for fold in manifest["folds"]:
        assert not set(fold["train_parts"]) & set(fold["test_parts"])
        train_parts = {item["part"] for item in fold["train"]}
        test_parts = {item["part"] for item in fold["test"]}
        assert not train_parts & test_parts


def test_mvtec_fit_and_cal_are_disjoint() -> None:
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    for category in ("carpet", "bottle", "capsule", "metal_nut", "transistor"):
        items = manifest["categories"][category]["items"]
        fit = {item["image"] for item in items if item["split"] == "fit"}
        cal = {item["image"] for item in items if item["split"] == "cal"}
        test = {item["image"] for item in items if item["split"] == "test"}
        assert fit and cal and test
        assert not fit & cal
        assert not fit & test
        assert not cal & test
