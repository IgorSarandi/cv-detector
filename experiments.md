- Oxford Pets box is a head, class cat_head

E0 Sphynx cat was predicted as dog 0.8

with lowering threshold for AP, our metrics grows, but model is the same. The problem is a mismatch of box definition, not in ability of quality searching cats.

| ID | What changed                                            | Data                     | mAP@0.5 | mAP@0.5:0.95  | P | R | Notes |
|----|---------------------------------------------------------|--------------------------|---------|---------------|---|---|-------|
| E0 | Pretrained COCO yolov8n, class cat (id 15), no training | val: 179 img / 180 boxes | 0.056   | 0.021 | 0.112 | 0.456 | Annotation mismatch: model draws whole-body box, GT is head box (IoU on 6 samples 0.17-0.45). Sphynx detected as dog 0.80. |
