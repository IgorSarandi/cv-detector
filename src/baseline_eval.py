"""E0, numeric part: evaluate pretrained COCO yolov8n on our val split.

COCO's "cat" has class id 15, ours has 0, so we make a copy of val with id 15.
"""
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

src = Path("data/cats_yolo")
dst = Path("data/cats_coco_ids")
if dst.exists():
    shutil.rmtree(dst)
(dst / "images/val").mkdir(parents=True)
(dst / "labels/val").mkdir(parents=True)

for img in (src / "images/val").glob("*.jpg"):
    shutil.copy2(img, dst / "images/val" / img.name)
    lines = (src / "labels/val" / f"{img.stem}.txt").read_text().splitlines()
    fixed = ["15 " + l.split(" ", 1)[1] for l in lines if l.strip()]   # replace class id
    (dst / "labels/val" / f"{img.stem}.txt").write_text("\n".join(fixed) + "\n")

model = YOLO("yolov8n.pt")
cfg = dst / "data.yaml"
cfg.write_text(yaml.safe_dump({"path": str(dst.resolve()), "train": "images/val", "val": "images/val",
                               "names": model.names}))

r = model.val(data=str(cfg), imgsz=640, batch=16, device="cpu", plots=False)
print("mAP@0.5      :", round(r.box.map50, 4))
print("mAP@0.5:0.95 :", round(r.box.map, 4))
print("precision    :", round(r.box.mp, 4), "  recall:", round(r.box.mr, 4))
