# Maize Leaf Disease Classification - Final Report

**Final model:** Model A - YOLO26n-cls  (best.pt)
**Test accuracy:** 95.05%  on 626 held-out images
**Target accuracy:** 96% - not reached
**Seed:** 42   **Classes:** Blight, Common_Rust, Gray_Leaf_Spot, Healthy

## Model Comparison (test split, sections 3-5)

| Model | Accuracy | Precision | Recall | F1 | Train time | Size | Inference |
|---|---|---|---|---|---|---|---|
| Model A - YOLO26n-cls | 94.57% | 93.29% | 92.93% | 93.10% | 1.9 min | 3.2 MB | 0.3 ms |
| Model B - Custom CNN | 89.94% | 87.92% | 87.48% | 87.62% | 4.7 min | 100.9 MB | 1.7 ms |

## Error Reduction (validation split, section 7)

| Configuration | Validation accuracy |
|---|---|
| baseline (section 5 winner) | 95.69% |
| exp_a_balanced_aug | 95.85% |
| exp_b_larger_model | 96.81% |

## Inference

```python
from ultralytics import YOLO
model = YOLO('models/maize_disease_yolo26n.pt')
result = model.predict('data/Healthy/Corn_Health (1).jpg')
```
