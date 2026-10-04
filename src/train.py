"""Fine-tune YOLO on the cat-head dataset (experiments E1, E2, ...).

Examples:
  # smoke test (a few minutes on CPU): checks that the whole pipeline works
  python src/train.py --name smoke --epochs 1 --imgsz 320 --fraction 0.1
  # E1: backbone frozen (first 10 layers of YOLOv8 = backbone)
  python src/train.py --name e1_frozen --epochs 30 --freeze 10
  # E2: everything trainable, smaller learning rate
  python src/train.py --name e2_full --epochs 50 --lr0 0.0002
"""
import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, default=Path("data/cats_yolo"))
p.add_argument("--weights", default="yolov8n.pt")
p.add_argument("--name", required=True, help="run name, results go to runs/train/<name>")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--freeze", type=int, default=0, help="freeze first N layers (10 = YOLOv8 backbone)")
p.add_argument("--optimizer", default="AdamW", help="explicit optimizer (the default 'auto' silently ignores --lr0)")
p.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
p.add_argument("--fraction", type=float, default=1.0, help="use only this share of train (smoke tests)")
p.add_argument("--patience", type=int, default=10, help="early stopping: epochs without val improvement")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-plots", action="store_true")
args = p.parse_args()

# Always write the yaml here, so absolute paths are right on every machine (local, Colab).
# The test split is deliberately NOT listed: we do not touch it until the very end.
data_dir = args.data_dir.resolve()

if not (data_dir / "images" / "train").is_dir():
    sys.exit(f"ERROR: {data_dir}/images/train not found. Run src/prepare_data.py first.")

cfg = data_dir / "train.yaml"
cfg.write_text(f"path: {data_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: cat_head\n")

model = YOLO(args.weights)          # nc=1 head is rebuilt automatically, matching weights are transferred
model.train(
    data=str(cfg), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    freeze=args.freeze if args.freeze > 0 else None,
    optimizer=args.optimizer, lr0=args.lr0, fraction=args.fraction, patience=args.patience,
    workers=args.workers, device=args.device, seed=args.seed,
    project=str(Path("runs/train").resolve()), name=args.name, plots=not args.no_plots,
)

# Final numbers on val, same protocol as E0.
best = Path(model.trainer.save_dir) / "weights" / "best.pt"
r = YOLO(str(best)).val(data=str(cfg), split="val", imgsz=args.imgsz, device=args.device, plots=False)
print(f"\nbest weights : {best}")
print("mAP@0.5      :", round(r.box.map50, 4))
print("mAP@0.5:0.95 :", round(r.box.map, 4))
print("precision    :", round(r.box.mp, 4), "  recall:", round(r.box.mr, 4))
