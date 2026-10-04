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

## Current step (3.0)
- [x] 1.1 prepare_data.py run: 831/179/178 images, 831/180/178 boxes
- [x] 1.2 check_labels.py: boxes are on cat heads
- [x] 2.1 Visual baseline: whole-body boxes, 5/6 found, sphynx -> dog 0.80
- [x] 2.2 E0 measured: mAP@0.5 0.056, mAP@0.5:0.95 0.021, P 0.112, R 0.456
- [x] 2.3 E0 recorded in experiments.md
- [ ] 3.0 Decide where to train (Colab vs local CPU), then E1: frozen backbone
