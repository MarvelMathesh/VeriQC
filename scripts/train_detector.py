#!/usr/bin/env python3
"""Train the supervised known-defect detector for one profile."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.datasets import NEU_CLASSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/prepared/neu/dataset.yaml")
    parser.add_argument("--init-weights", default="models/weights/yolo26n.pt")
    parser.add_argument("--arch", default="yolo26n")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--out", default="models/profiles/steel_surface/yolo26n_640.pt")
    parser.add_argument("--report", default="reports/tables/detector_neu.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    data = str(REPO / args.data)
    init = Path(args.init_weights)
    if not init.is_absolute():
        init = REPO / init
    if not init.is_file():
        raise SystemExit(
            f"initial detector checkpoint not found: {init}. "
            "Place the COCO-initialised yolo26n.pt there instead of relying "
            "on an implicit download."
        )
    init_hash = sha256_file(init)
    started = time.time()
    model = YOLO(str(init))
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        patience=max(20, args.epochs // 4),
        workers=4,
        project=str(REPO / "runs" / "detect"),
        name=f"{args.arch}_{args.imgsz}",
        exist_ok=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        verbose=True,
    )
    metrics = model.val(data=data, split="test", imgsz=args.imgsz, device=args.device, verbose=False)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    best = Path(model.trainer.best)
    import shutil

    shutil.copy2(best, out)
    payload = {
        "profile_id": "steel_surface",
        "architecture": args.arch,
        "input_size": args.imgsz,
        "epochs": args.epochs,
        "seed": args.seed,
        "classes": list(NEU_CLASSES),
        "init_weights": str(init.relative_to(REPO)) if init.is_relative_to(REPO) else str(init),
        "init_weights_sha256": init_hash,
        "weights": str(out.relative_to(REPO)),
        "weights_sha256": sha256_file(out),
        "train_seconds": round(time.time() - started, 1),
        "test": {
            "split": "official NEU-DET validation",
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "command": " ".join([sys.executable, *sys.argv]),
    }
    report = REPO / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
