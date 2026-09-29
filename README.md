# Maize Leaf Disease Classification with YOLO26

Classify a maize leaf image into one of four classes — **Healthy**, **Common Rust**, **Gray Leaf Spot**, **Blight** — using a YOLO26 classification model.

## Problem Statement

Maize foliar diseases spread rapidly and can cut yields by 30–50%. The current diagnosis method is visual inspection by a human agronomist: slow, subjective, error-prone, and impossible to scale across large plantations. Many smallholder farmers have no access to a plant pathologist, so disease is caught only after irreversible damage.

**Objective:** classify a maize leaf image into one of four classes so a farmer can photograph a leaf and get an instant diagnosis.

## Research Questions

1. How accurately can a pretrained YOLO26 classifier separate these four classes?
2. Which classes get confused with each other, and why — background clutter, lighting, or the visual overlap between Blight and Gray Leaf Spot?
3. How do model size, augmentation, and training time trade off against accuracy on a dataset this small?

## Project Overview

A supervised image-classification pipeline:

```
Data → Validation → 80/20 Split → YOLO26 Classification → Evaluation
     → Inference → Error Analysis → Error Reduction → Export
```

This is a **classification** problem, not a detection one: the dataset gives one label per photo and contains no bounding boxes, so the model is asked *what disease is this* rather than *where on the leaf is the lesion*.

- **Model:** `yolo26n-cls.pt`, the smallest and fastest YOLO26 classification model.
- **Hardware:** Apple M4 GPU (MPS). No CUDA on this machine.
- **Speed:** `imgsz=256` (3,852 of 4,188 images are already 256×256), `batch=64` on the GPU, and no RAM image cache — which was measured to be *slower* here.
- **Runtime:** ~2–2.5 minutes for sections 0–6, plus ~10 minutes for the two retraining experiments in section 7.

## Dataset Overview

**Source:** [Kaggle — Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghouse/corn-or-maize-leaf-disease-dataset)

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.7% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 574 | 13.7% |
| **Total** | **4,188** | **100%** |

Real-world leaf photographs stored as `data/<Class_Name>/*.jpg`, where the folder name *is* the label. The split is 80/20 (3,352 train / 836 validation). The dataset is imbalanced: Gray Leaf Spot has 2.3× fewer images than Common Rust, and it is also the class the model struggles with most.

**Known issues in the raw data:**

- Mixed file extensions (`.jpg`, `.JPG`, `.jpeg`) — 501 validation images end in uppercase `.JPG`, which a lowercase `*.jpg` glob silently misses.
- Mixed colour modes: 4 images are RGBA and 1 is CMYK.
- Resolutions from 180×116 up to 5184×5184, though 3,852 images are exactly 256×256.
- Non-uniform backgrounds and lighting, plus possible label noise where harmless blemishes resemble early-stage disease.

## Project Structure

```
maize_disease/
├── data/                              # source images, one folder per class
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── models/                            # pretrained + trained weights
│   ├── yolo26n-cls.pt                 # pretrained baseline
│   ├── yolo26s-cls.pt                 # larger model (section 7.5)
│   └── maize_disease_yolo26n.pt       # trained model, ready to use
├── Maize_Disease_Detection_YOLO.ipynb # the full 0-8 pipeline
├── .gitignore
└── README.md
```

Generated during a run and safe to delete:

- `data_cls/` — 80/20 train/val split created by notebook section 2
- `data_balanced/` — oversampled training set created by section 7.2
- `runs/classify/` — training logs, plots, confusion matrices, checkpoints

## Notebook Structure

- **0. Environment Setup & Configuration**
  - 0.1 Install Dependencies
  - 0.2 Import Libraries
  - 0.3 Configuration
- **1. Data Preparation & Validation**
  - 1.1 Dataset Paths
  - 1.2 Dataset Validation
  - 1.3 Class Inspection
- **2. Create Dataset Folder Structure**
  - 2.1 Generate Folder Structure & Split
  - 2.2 Verify Split
- **3. Model Initialization & Training**
  - 3.1 Load Pre-trained Classification Model
  - 3.2 Configure Training Arguments
  - 3.3 Start Training
  - 3.4 Visualize Training Results
- **4. Model Evaluation & Validation**
  - 4.1 Run Validation
  - 4.2 Display Metrics
  - 4.3 Confusion Matrix
- **5. Inference & Visual Inspection**
  - 5.1 Single Image Prediction
  - 5.2 Batch Prediction
  - 5.3 Manual QA
  - 5.4 Predict on a Single Custom Image
- **6. Error Analysis**
  - 6.1 Collect Misclassified Images
  - 6.2 Per-Class Error Rate
  - 6.3 Top Confusion Pairs
  - 6.4 Confidence Distribution of Errors
  - 6.5 Visualize Misclassified Samples
  - 6.6 Identify Error Patterns (lighting, blur, background, etc.)
- **7. Error Reduction**
  - 7.1 Data Augmentation (flip, rotate, color jitter, blur)
  - 7.2 Class Balancing (oversample, undersample, class weights)
  - 7.3 Add More Data for Weak Classes
  - 7.4 Hyperparameter Tuning (lr, epochs, batch size, img size)
  - 7.5 Try Larger Model (yolo26s-cls, yolo26m-cls)
  - 7.6 Re-train & Compare Metrics
  - 7.7 Iterate Until Target Accuracy Reached
- **8. Model Export & Deployment (Optional)**
  - 8.1 Export Model
  - 8.2 Save Model Weights
