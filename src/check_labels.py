"""Draw YOLO boxes on random images to verify the conversion visually.

Usage: python src/check_labels.py --data-dir data/cats_yolo --split val --n 6
"""
import argparse
import random
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, default=Path("data/cats_yolo"))
p.add_argument("--split", default="train")
p.add_argument("--n", type=int, default=6)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

labels = sorted((args.data_dir / "labels" / args.split).glob("*.txt"))
random.Random(args.seed).shuffle(labels)
out_dir = args.data_dir / "check"
out_dir.mkdir(exist_ok=True)

for lab in labels[: args.n]:
    img = cv2.imread(str(args.data_dir / "images" / args.split / f"{lab.stem}.jpg"))
    h, w = img.shape[:2]
    for line in lab.read_text().split("\n"):
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)   # de-normalize
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{lab.stem}.jpg"), img)
    print("saved", out_dir / f"{lab.stem}.jpg")
