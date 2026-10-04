# Plan

Goal: cat-head detector with YOLO, then robot vacuum, then torchvision, then geometry, then mock interview.

## Stages
- [ ] 1. Data: prepare_data.py -> train/val/test split
- [ ] 2. Baseline E0: pretrained model on val
- [ ] 3. Fine-tuning E1 (head only), E2 (unfreeze)
- [ ] 4. Error analysis
- [ ] 5. Robot vacuum project
- [ ] 6. Faster R-CNN in torchvision
- [ ] 7. Camera geometry
- [ ] 8. Mock interview

## Current step (3.4)
- [x] 1.1 prepare_data.py run: 831/179/178 images, 831/180/178 boxes
- [x] 1.2 check_labels.py: boxes are on cat heads
- [x] 2.1 Visual baseline: whole-body boxes, 5/6 found, sphynx -> dog 0.80
- [x] 2.2 E0 measured: mAP@0.5 0.056, mAP@0.5:0.95 0.021, P 0.112, R 0.456
- [x] 2.3 E0 recorded in experiments.md
- [x] 3.0 Decide where to train (Colab vs local CPU), then E1: frozen backbone
- [x] 1, 2 done (data, E0)
- [x] 3.3 E1a done locally: mAP@0.5 0.995, mAP@0.5:0.95 0.907, P 0.998, R 0.994 (imgsz 320, freeze 10, val)
- [x] 3.4 iou_report for E1 and E0 done (E1: mean IoU 0.937; E0: 0.249). Colab was on CPU; stopped
- [ ] 3.5 Ablation: E0 weights at imgsz 320, E1 weights at imgsz 640 (write predictions first)
- [ ] 3.6 Record E1a in experiments.md; sanity checks (test split once, stress photos)
