- Oxford Pets box is a head, class cat_head

E0 Sphynx cat was predicted as dog 0.8

with lowering threshold for AP, our metrics grows, but model is the same. The problem is a mismatch of box definition, not in ability of quality searching cats.

| ID | What changed                                            | Data                     | mAP@0.5 | mAP@0.5:0.95  | P | R | Notes |
|----|---------------------------------------------------------|--------------------------|---------|---------------|---|---|-------|
| E0 | Pretrained COCO yolov8n, class cat (id 15), no training | val: 179 img / 180 boxes | 0.056   | 0.021 | 0.112 | 0.456 | Annotation mismatch: model draws whole-body box, GT is head box (IoU on 6 samples 0.17-0.45). Sphynx detected as dog 0.80. |
| E1a | fine-tune yolov8n.pt, freeze=10 (backbone), AdamW lr0=0.001, imgsz 320, seed 42, CPU | train 831 / val 179 (180 boxes) | 0.995 | 0.907 | 0.998 | 0.994 | iou_report: all 180 GT found with IoU >= 0.65, mean best IoU 0.937. Worst: Persian_217 0.66, Bengal_105 0.70 (two cats). Val is saturated; test not evaluated yet. Epochs run: see results.csv |
| A1 | E0 weights, imgsz 320 instead of 640 | val | share@0.5 11.1% (was 10.6%), approx mAP@0.5:0.95 0.037 (was 0.030) | image size is NOT the cause of the E0→E1 jump |
| A2 | E1 weights, imgsz 640 instead of 320 | val | share@0.5 96.1% (was 100%), mean IoU 0.878 (was 0.937), approx mAP@0.5:0.95 0.841 (was 0.924) | train/inference scale mismatch: ~7 heads lost, remaining boxes less tight. Hypothesis, not tested at 256/480 |



