"""E0, visual part: run pretrained COCO yolov8n on the same images that check_labels.py drew."""
from pathlib import Path

from ultralytics import YOLO

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0
    
def read_gt(img_path):          # YOLO txt -> list of (x1, y1, x2, y2), normalized 0..1
    lab = img_path.parents[2] / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
    boxes = []
    for line in lab.read_text().split("\n"):
        if line.strip():
            _, xc, yc, w, h = map(float, line.split())
            boxes.append((xc - w/2, yc - h/2, xc + w/2, yc + h/2))
    return boxes

data = Path("data/cats_yolo")
stems = [p.stem for p in (data / "check").glob("*.jpg")]
imgs = [next((data / "images").glob(f"*/{s}.jpg")) for s in stems]   # originals, without drawn boxes

model = YOLO("yolov8n.pt")          # weights are downloaded on the first run

out = str(Path("runs").resolve())
for p in imgs:
    r = model.predict(str(p), classes=[15], conf=0.25,   # 15 = "cat" in COCO
                        save=True, project="runs", name="e0_visual", exist_ok=True)[0]
    gt = read_gt(p)
    for pb in r.boxes.xyxyn.tolist():      # predicted boxes, normalized xyxy
        print("   IoU with GT:", [round(iou(pb, g), 2) for g in gt])
    print(p.name, [round(float(c), 2) for c in r.boxes.conf])

#r = model.predict(str(imgs[2]), conf=0.1, save=True, project="runs", name="e0_visual_1", exist_ok=True)[0]
#print([model.names[int(c)] for c in r.boxes.cls])
