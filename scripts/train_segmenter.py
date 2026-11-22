#!/usr/bin/env python3
"""Train the crack segmenter on the official KolektorSDD folds."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.datasets import (  # noqa: E402
    load_manifest,
    split_by_part,
    split_kolektor_fixed,
    split_kolektor_parts,
)
from veriqc.models.segmenter import Segmenter, SegmenterConfig, UNet  # noqa: E402

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id: int) -> None:
    base = torch.initial_seed() % 2**32
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


class KolektorDataset(Dataset):
    def __init__(self, items: list[dict], width: int, height: int, train: bool) -> None:
        self.items = items
        self.width = width
        self.height = height
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item["image"]) as opened:
            image = opened.convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        with Image.open(item["label"]) as opened:
            mask = opened.convert("L").resize((self.width, self.height), Image.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
        if self.train:
            if random.random() < 0.5:
                image_array = image_array[:, ::-1].copy()
                mask_array = mask_array[:, ::-1].copy()
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.04, 0.04)
            image_array = np.clip(image_array * gain + bias, 0.0, 1.0)
        image_array = (image_array - _MEAN) / _STD
        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        target = torch.from_numpy(mask_array).unsqueeze(0)
        return tensor, target


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    union = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def split_items(items: list[dict], seed: int, mode: str) -> tuple[list[dict], list[dict], list[dict]]:
    if mode == "nested":
        return split_kolektor_parts(items, seed)
    if mode == "fixed":
        return split_kolektor_fixed(items, seed)
    train, val = split_by_part(items, seed)
    return train, val, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="-1 trains every fold")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--split", choices=["current", "nested", "fixed"], default="current")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="models/profiles/commutator")
    parser.add_argument("--report", default="reports/tables/segmenter_kolektor_training.json")
    args = parser.parse_args()

    manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
    folds = [int(args.fold)] if args.fold >= 0 else list(range(len(manifest["folds"])))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results = []
    for fold_index in folds:
        fold = manifest["folds"][fold_index]
        fold_seed = args.seed + fold_index
        seed_everything(fold_seed)
        train_items, val_items, cal_items = split_items(fold["train"], fold_seed, args.split)
        train_dataset = KolektorDataset(train_items, args.width, args.height, train=True)
        val_dataset = KolektorDataset(val_items, args.width, args.height, train=False)
        loader_generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            worker_init_fn=worker_init,
            generator=loader_generator,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=max(1, args.workers // 2),
                pin_memory=device.type == "cuda",
            )
            if val_items
            else None
        )

        model = UNet(pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        positive_weight = torch.tensor([10.0], device=device)
        best_loss = float("inf") if val_loader is not None else None
        best_epoch = 0
        started = time.time()
        checkpoint = REPO / args.out / f"fold{fold_index}_unet.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
                    loss = bce + dice_loss(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item())
            scheduler.step()

            train_loss = running / max(1, len(train_loader))
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(device)
                        targets = targets.to(device)
                        logits = model(images)
                        bce = F.binary_cross_entropy_with_logits(
                            logits, targets, pos_weight=positive_weight
                        )
                        val_loss += float((bce + dice_loss(logits, targets)).item())
                val_loss /= max(1, len(val_loader))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    Segmenter(
                        model,
                        SegmenterConfig(args.width, args.height),
                        device=str(device),
                    ).save(checkpoint)
                print(
                    f"fold {fold_index} epoch {epoch:03d} train {train_loss:.4f} "
                    f"val {val_loss:.4f}",
                    flush=True,
                )
            else:
                # Fixed-budget arm: the final epoch is the pre-registered
                # checkpoint, so calibration parts are never used for selection.
                best_epoch = epoch
                Segmenter(
                    model,
                    SegmenterConfig(args.width, args.height),
                    device=str(device),
                ).save(checkpoint)
                print(
                    f"fold {fold_index} epoch {epoch:03d} train {train_loss:.4f}",
                    flush=True,
                )

        if not checkpoint.exists():
            raise RuntimeError(f"no checkpoint was written for fold {fold_index}")
        results.append(
            {
                "fold": fold_index,
                "input_size": [args.width, args.height],
                "split": args.split,
                "seed": fold_seed,
                "epochs": args.epochs,
                "train_parts": sorted({str(item["part"]) for item in train_items}),
                "val_parts": sorted({str(item["part"]) for item in val_items}),
                "cal_parts": sorted({str(item["part"]) for item in cal_items}),
                "calibration_parts": (
                    sorted({str(item["part"]) for item in val_items})
                    if args.split == "current"
                    else sorted({str(item["part"]) for item in cal_items})
                ),
                "calibration_shares_validation": args.split == "current",
                "train_images": len(train_items),
                "val_images": len(val_items),
                "cal_normal_images": sum(1 for item in cal_items if not item["is_defect"]),
                "best_epoch": best_epoch,
                "best_val_loss": round(best_loss, 6) if best_loss is not None else None,
                "train_seconds": round(time.time() - started, 1),
                "weights": display_path(checkpoint),
                "weights_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    report = REPO / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"folds": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
