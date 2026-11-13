"""Reproducible detector training for the combined NEU-DET + KolektorSDD set.

Every candidate in the E4 bake-off is trained by this script with the same
recipe, so no candidate can win because it was trained more carefully.  The
recipe is fixed up front:

* identical data split (the combined 7-class dataset, part-wise for
  KolektorSDD),
* identical optimiser family, epoch budget and early-stopping patience,
* identical seed,
* the only differences are the architecture, the input resolution and whether
  the initialisation is the COCO-pretrained weights or random.

``--init scratch`` exists so the contribution of the pretrained initialisation
is *measured* rather than assumed.  It is the same question the proposal asks
of the anomaly gate: how much of the result is the recipe and how much is
inherited knowledge.

Usage::

    python scripts/train_detector.py --arch yolo26n --imgsz 640 \
        --data data/combined_yolo/dataset.yaml --out models/ultralytics/yolo26n_c640
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "tables"


def train_ultralytics(args) -> dict:
    """YOLO26 / YOLO11 through the Ultralytics trainer."""
    from ultralytics import YOLO

    if args.init == "scratch":
        # the yaml, not the .pt: no pretrained weights at all
        model = YOLO(f"{args.arch}.yaml")
        init_desc = "scratch (random init)"
    else:
        w = Path(args.init_weights or (REPO / "models" / "weights" / f"{args.arch}.pt"))
        if not w.exists():
            raise SystemExit(f"missing init weights: {w}")
        model = YOLO(str(w))
        init_desc = f"COCO-pretrained ({w.name})"

    t0 = time.perf_counter()
    res = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(REPO / "models" / "ultralytics"),
        name=args.out_name,
        exist_ok=True,
        verbose=False,
        plots=False,
        pretrained=False if args.init == "scratch" else True,
    )
    seconds = time.perf_counter() - t0

    out_dir = REPO / "models" / "ultralytics" / args.out_name
    best = out_dir / "weights" / "best.pt"
    metrics = {}
    try:
        metrics = {k: float(v) for k, v in (res.results_dict or {}).items()}
    except Exception:
        pass
    return {
        "arch": args.arch,
        "family": "ultralytics",
        "imgsz": args.imgsz,
        "init": init_desc,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "seed": args.seed,
        "train_seconds": round(seconds, 1),
        "weights": str(best),
        "weights_bytes": best.stat().st_size if best.exists() else None,
        "val_metrics": metrics,
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="yolo26n")
    ap.add_argument("--data", default="data/combined_yolo/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()

    if args.data.startswith("data/"):
        args.data = str(REPO / args.data)

    info = train_ultralytics(args)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"train_{args.out_name}.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
