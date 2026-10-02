# Maize Leaf Disease Classification - Final Report

**Final model:** Model A - YOLO26s-cls (larger variant)  (best.pt)
**Test accuracy:** 95.68%  on 625 held-out images, scored once
**Target accuracy:** 96% - not reached on test
**Seed:** 42   **Classes:** Blight, Common_Rust, Gray_Leaf_Spot, Healthy

## Model Comparison (test split, sections 3-5)

| Model | Val F1 | Accuracy | Precision | Recall | F1 | Train time | Size | Inference |
|---|---|---|---|---|---|---|---|---|
| Model A - YOLO26n-cls | 93.75% | 94.56% | 92.76% | 93.99% | 93.27% | 3.7 min | 3.2 MB | 0.2 ms |
| Model B - Custom CNN | 88.65% | 89.44% | 87.03% | 86.88% | 86.83% | 7.2 min | 100.9 MB | 2.6 ms |

## Error Reduction (validation split, section 7)

| Configuration | Val macro F1 | Val accuracy |
|---|---|---|
| baseline (section 5 winner) | 93.75% | 95.04% |
| exp_a_augmentation | 94.68% | 95.84% |
| exp_b_larger_model | 95.01% | 96.00% |

## Limitations

- 625 held-out images from one public compilation (PlantVillage + PlantDoc). No field-trial
  data, no geographic or seasonal metadata. The score does not predict performance on a new farm.
- Roughly +/-1.7 points of sampling uncertainty at this test size. Differences under about 3 points
  are not reliably distinguishable.
- Class imbalance was fixed by repeating training images. Gray Leaf Spot still has only 572
  distinct photographs underneath.
- Label noise is present: two raw photographs carried two different class labels at once.
- No severity, growth stage, or lesion location. Grad-CAM is a visualisation, not a measurement.
- Intended use is triage and prioritisation with agronomist confirmation. Not standalone diagnosis,
  not automated treatment decisions, not validated outside this dataset.

## Inference

```python
from ultralytics import YOLO
model = YOLO('models/maize_disease_yolo26s.pt')
result = model.predict('data/Healthy/Corn_Health (1).jpg')
```
