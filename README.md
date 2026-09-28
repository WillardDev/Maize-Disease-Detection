# Maize Leaf Disease Detection using CNN

## Problem Definition

Maize is heavily threatened by foliar diseases — Common Rust, Gray Leaf Spot and Blight — which
spread fast and can cut yields by 30–50%. The current diagnosis method is visual inspection by
human agronomists, which is slow, subjective, error-prone and impossible to scale across large
plantations. Many smallholder farmers have no access to a plant pathologist, so disease is
detected only after irreversible damage. This project replaces manual inspection with automatic,
image-based diagnosis.

## Project Objective

Train a Convolutional Neural Network to classify a maize leaf image into one of four classes —
**Healthy, Common Rust, Gray Leaf Spot, Blight** — so a farmer can photograph a leaf and get an
instant diagnosis.

Objectives:
1. Build a CNN classifier for the 4-class problem.
2. Clean and augment the data to handle noisy, real-world field images.
3. Compare CNN architectures and select the best performer.
4. Analyse errors and reduce them.
5. Deploy a simple prediction tool.

## Project Overview

A supervised image-classification pipeline:

```
Data → Cleaning → EDA → Train/Val/Test Split → Augmentation → CNN Training
     → Evaluation → Error Analysis → Error Reduction → Prediction
```

A CNN learns features directly from raw pixels — edges and colour blobs in early layers,
lesion shapes and textures in deeper ones — so no hand-crafted features are needed. The
dataset is imbalanced (Gray Leaf Spot has ~43% fewer images than Common Rust), so class
weights and macro-averaged metrics are used so the model can't just favour majority classes.

## Research Questions

1. Which CNN architecture (custom CNN, VGG16, ResNet50, InceptionV3, MobileNetV2) gives the
   highest accuracy, and how do model size and training time trade off against it?
2. How do augmentation and class-balancing (rotation, flips, colour jitter, class weights)
   affect generalisation, especially recall on the minority Gray Leaf Spot class?
3. Which disease pairs are most often confused, and what causes it — background clutter,
   lighting, or the visual overlap between Common Rust and Blight?

## Data Source

<https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset>

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.7% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 574 | 13.7% |
| **Total** | **4,188** | **100%** |

Real-world leaf photographs (JPEG/JPG). Issues in the raw data: mixed file extensions
(`.jpg`/`.JPG`/`.jpeg`), varying resolutions, non-uniform backgrounds and lighting, possible
duplicate captures, and label noise where harmless blemishes resemble early-stage disease.
Stored locally as `data/<Class_Name>/*.jpg`.

## Tools

Python 3 · Jupyter Notebook · NumPy / pandas · PyTorch / torchvision · OpenCV / Pillow ·
Matplotlib / Seaborn · scikit-learn

## Project Structure

```
maize_disease/
├── data/                          # Dataset (4 class folders)
│   ├── Blight/                   #   1,146 images
│   ├── Common_Rust/              #   1,306 images
│   ├── Gray_Leaf_Spot/           #     574 images
│   └── Healthy/                  #   1,162 images
├── Maize_Disease_Detection.ipynb  # End-to-end pipeline
├── README.md                      # Documentation
└── requirements.txt               # Dependencies
```

Labels come from folder names via `ImageFolder`, so new images dropped into a folder are
picked up automatically.

## Notebook Structure

1. **Problem statement**
2. **Data loading**
3. **Data cleaning**
4. **EDA**
5. **Train/val/test split**
6. **Data augmentation**
7. **Model training**
8. **Evaluation**
9. **Error analysis**
10. **Error reduction**
11. **Save & predict**
12. **Conclusion & future work**

## Methodology: Neural Networks (Supervised — Classification)

**When to use:** 
- Large datasets with highly non-linear patterns, especially unstructured data
(images, text, audio) where hand-crafted features are hard to design.

**Dataset needed:** 
- Large volume ideally, as numeric tensors (images as pixel arrays). 4,188
images is moderate — workable because transfer learning starts from ImageNet features instead of
learning vision from scratch, but not enough for training a deep CNN from zero.

**Cleaning & why:** 
- Discard corrupt files that break loading; remove duplicates that would put the same leaf in train and test
- Scale/normalise inputs
- Resize to a consistent 224×224
- Augment to grow the effective dataset size and build invariance to how leaves are photographed.

**EDA & why:** 
- Confirm the volume is sufficient (it determines the modelling approach); 
- Check class balance, which exposes the 13.7% vs 31.2% imbalance driving class weights; 
- Sample images to sanity-check labels and quality; per-class colour statistics confirm the classes are separable.

**Train/test split:** 
- Stratified train/validation/test, with a dedicated validation set for early
stopping — not just CV, since training is expensive.

**Model training:** 
- Forward pass computes predictions, the loss measures error, backpropagation
computes gradients, and an optimiser (Adam) updates weights over many epochs.

**Evaluation metrics:** 
- Classification — accuracy, precision, recall, F1, ROC-AUC, plus
train/validation loss curves.

**Error analysis & why:** 
- Train vs. validation curves catch overfitting and underfitting
- Inspecting misclassified examples shows whether labels, augmentation or the architecture need
fixing.

**Conclusion:** 
- A CNN is the right tool for this problem because leaf lesions are complex,
non-linear visual patterns that no hand-written rule can capture, and the model learns the
features itself. With 4,188 images, transfer learning from a pretrained backbone is the practical
route to high accuracy rather than training a deep network from scratch.

**Way Forward:**
- Deploy as a mobile app with on-device (edge/TFLite) inference so farmers get diagnoses offline.
- Extend the dataset with more diseases, other crops, and images from real field conditions
  across different seasons and regions.
- Add explainability (Grad-CAM heatmaps) to show *which* leaf region drove the prediction,
  addressing the black-box limitation and building farmer trust.
- Add severity grading and treatment recommendations, turning classification into actionable
  guidance.
- Track field performance over time and retrain on newly collected data to handle drift.
