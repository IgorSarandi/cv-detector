from ultralytics import YOLO

r = YOLO("runs/train/e1_cpu320/weights/best.pt").val(
    data="data/cats_yolo/data.yaml", split="test", imgsz=320, device="cpu", plots=False)
print("mAP@0.5      :", round(r.box.map50, 4))
print("mAP@0.5:0.95 :", round(r.box.map, 4))
print("precision    :", round(r.box.mp, 4), "  recall:", round(r.box.mr, 4))
