"""Per-image IoU report: how good are the predicted boxes, and how does that turn into mAP?

Usage:
  python src/iou_report.py --weights runs/train/e1_cpu320/weights/best.pt --imgsz 320
  python src/iou_report.py --weights yolov8n.pt --cls 15 --imgsz 640      # E0: COCO 'cat' has id 15

Use the SAME --imgsz as in training/validation, otherwise the numbers will differ.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]          # 0.50, 0.55, ..., 0.95


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_gt(label_path):
    """YOLO txt -> list of (x1, y1, x2, y2), normalized 0..1."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            _, xc, yc, w, h = map(float, line.split())
            boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def summarize(ious):
    """Share of ground-truth boxes with IoU >= t, for every threshold t, and the mean of those shares."""
    shares = [sum(i >= t for i in ious) / len(ious) for t in THRESHOLDS]
    return shares, sum(shares) / len(shares)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/cats_yolo"))
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--cls", type=int, default=None, help="keep only this predicted class id (15 for COCO cat)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--worst", type=int, default=5, help="how many worst images to list")
    args = p.parse_args()

    model = YOLO(args.weights)
    results = []                                  # (iou, image name) for every GT box
    n_pred = 0
    for img in sorted((args.data_dir / "images" / args.split).glob("*.jpg")):
        gts = read_gt(args.data_dir / "labels" / args.split / f"{img.stem}.txt")
        r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, device=args.device,
                          classes=[args.cls] if args.cls is not None else None, verbose=False)[0]
        preds = r.boxes.xyxyn.tolist()
        n_pred += len(preds)
        for g in gts:
            results.append((max((iou(pr, g) for pr in preds), default=0.0), img.name))

    ious = [v for v, _ in results]
    shares, mean_share = summarize(ious)
    print(f"{len(ious)} ground-truth boxes, {n_pred} predicted boxes (conf >= {args.conf})")
    print(f"mean best IoU per box: {sum(ious) / len(ious):.3f}\n")
    print("threshold t | share of GT boxes with IoU >= t")
    for t, s in zip(THRESHOLDS, shares):
        print(f"   {t:.2f}     | {s:6.1%}  {'#' * round(s * 40)}")
    print(f"\napprox mAP@0.5      = share at 0.50          = {shares[0]:.3f}")
    print(f"approx mAP@0.5:0.95 = mean of the 10 shares  = {mean_share:.3f}")
    print("(an idealisation: every box found once, perfect ranking, no extra boxes;")
    print(" compare with the official numbers from val() - the gap is what this model does NOT get right)")
    print(f"\n{args.worst} worst images:")
    for v, name in sorted(results)[: args.worst]:
        print(f"   IoU {v:.2f}  {name}")


if __name__ == "__main__":
    main()
