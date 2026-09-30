# Maize Leaf Disease Classification: YOLO26 vs Custom CNN

Classify a maize leaf image into one of four classes — **Healthy**, **Common Rust**, **Gray Leaf Spot**, **Blight** — and compare two models trained on the same split: a pretrained YOLO26 classifier and a custom CNN built from scratch in Keras.

## Problem Statement

Maize foliar diseases spread rapidly and can cut yields by 30–50%. The current diagnosis method is visual inspection by a human agronomist: slow, subjective, error-prone, and impossible to scale across large plantations. Many smallholder farmers have no access to a plant pathologist, so disease is caught only after irreversible damage.

**Objective:** classify a maize leaf image into one of four classes so a farmer can photograph a leaf and get an instant diagnosis.

## Research Questions

1. How accurately can a pretrained YOLO26 classifier separate these four classes, and how much of that is thanks to transfer learning rather than training from scratch?
2. How close does a small custom CNN get, and is it small and fast enough to be worth deploying instead?
3. Which classes get confused with each other, and why — background clutter, lighting, or the visual overlap between Blight and Gray Leaf Spot?
4. How do model size, augmentation, and training time trade off against accuracy on a dataset this small?

## Project Overview

A supervised image-classification pipeline with **two models** and a head-to-head comparison:

```
Data → Validation → 70/15/15 Split
    → Model A: YOLO26n-cls (transfer learning)  ─┐
    → Model B: Custom CNN (Keras, from scratch)  ─┴→ Comparison → best model
    → Inference → Error Analysis → Error Reduction → Export
```

This is a **classification** problem, not a detection one: the dataset gives one label per photo and contains no bounding boxes, so the model is asked *what disease is this* rather than *where on the leaf is the lesion*.

| | Model A | Model B |
|---|---|---|
| Model | `yolo26n-cls.pt` | custom CNN, 2 conv/pool blocks |
| Starting weights | ImageNet-pretrained | random |
| Framework | PyTorch (ultralytics) | TensorFlow / Keras |
| Why it is here | the accuracy benchmark | shows what transfer learning is actually worth |

Model A starts from ImageNet-pretrained weights, so it only has to learn four leaf classes rather than vision from scratch. That is why a handful of epochs is enough. Model B has to learn everything, which makes it slower and smaller but tells you how much the pretraining is really worth.

## Dataset Overview

**Source:** [Kaggle — Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghouse/corn-or-maize-leaf-disease-dataset)

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.7% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 574 | 13.7% |
| **Total** | **4,188** | **100%** |

Real-world leaf photographs stored as `data/<Class_Name>/*.jpg`, where the folder name *is* the label. The split is **70 / 15 / 15** (2,936 train / 626 validation / 626 test), and each part has one job:

- **train** — what both models learn from.
- **val** — loss curves every epoch, and the score for the section 7 experiments.
- **test** — used only for the final comparison and the report. Nothing is tuned on it, which is what makes the section 5 comparison fair.

The dataset is imbalanced: Gray Leaf Spot has 2.3× fewer images than Common Rust, and it is also the class the model struggles with most.

**Known issues in the raw data:**

- Mixed file extensions (`.jpg`, `.JPG`, `.jpeg`) — several hundred files end in uppercase `.JPG`, which a lowercase `*.jpg` glob silently misses.
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
│   ├── yolo26n-cls.pt                 # Model A baseline
│   ├── yolo26s-cls.pt                 # larger variant (section 7.5)
│   ├── maize_disease_yolo26n.pt       # trained YOLO model, ready to use
│   └── maize_disease_cnn.keras        # trained CNN (only if it won section 5.7)
├── Maize_Disease_Detection_YOLO.ipynb # the full 0-9 pipeline
├── .gitignore
└── README.md
```

Generated during a run and safe to delete:

- `data_cls/` — the 70/15/15 train/val/test split created by notebook section 2
- `data_balanced/` — oversampled training set created by section 7.2
- `runs/classify/`, `runs/cnn/` — training logs, plots, confusion matrices, checkpoints
- `model_comparison.md` — the final report written by section 9.3

## Requirements

```
ultralytics      # Model A
tensorflow       # Model B
opencv-python    # image reading
scikit-learn     # confusion matrices and classification reports
```

Section 0.1 installs the first two if they are missing. `tensorflow` pulls in Keras, and on macOS it runs on the CPU — there is no MPS support, which is a large part of why Model B trains slower than Model A.

## Notebook Structure

- **Environment Setup & Configuration**
  - Install Dependencies
  - Import Libraries
  - Configuration
  - Set Random Seeds (reproducibility)
- **1. Data Preparation & Validation**
  - 1.1 Dataset Paths
  - 1.2 Dataset Validation
  - 1.3 Class Inspection
  - 1.4 Class Distribution Check
- **2. Create Dataset Folder Structure**
  - 2.1 Generate Folder Structure & Split
  - 2.2 Verify Split
- **3. Model A: YOLO26n-cls Pipeline**
  - 3.1 Load Pre-trained Classification Model
  - 3.2 Configure Training Arguments
  - 3.3 Start Training
  - 3.4 Visualize Training Results
  - 3.5 Run Validation
  - 3.6 Display Metrics
  - 3.7 Confusion Matrix
  - 3.8 YOLO Metrics
- **4. Model B: Custom CNN Pipeline**
  - 4.1 Build Data Generators (train/val/test)
  - 4.2 Define CNN Architecture (Conv → Pool → Conv → Pool → Dense → Softmax)
  - 4.3 Compile Model (optimizer, loss, metrics)
  - 4.4 Set Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
  - 4.5 Train CNN
  - 4.6 Visualize Training Curves (loss, accuracy)
  - 4.7 Evaluate on Test Set
  - 4.8 Confusion Matrix
  - 4.9 Classification Report (precision, recall, F1)
  - 4.10 CNN Metrics
- **5. Model Comparison**
  - 5.1 Side-by-Side Metrics Table (Accuracy, Precision, Recall, F1)
  - 5.2 Training Time Comparison
  - 5.3 Model Size Comparison
  - 5.4 Inference Speed Comparison
  - 5.5 Accuracy vs Speed Trade-off Plot
  - 5.6 Confusion Matrix Comparison
  - 5.7 Select Best Model
- **6. Error Analysis on the Best Model**
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
  - 7.5 Try Larger Variant (yolo26s-cls OR deeper CNN)
  - 7.6 Re-train & Compare Metrics
  - 7.7 Iterate Until Target Accuracy Reached
- **8. Inference & Visual Inspection (Final Model)**
  - 8.1 Single Image Prediction
  - 8.2 Batch Prediction
  - 8.3 Manual QA
  - 8.4 Predict on a Single Custom Image
- **9. Model Export & Deployment**
  - 9.1 Export Model
  - 9.2 Save Model Weights
  - 9.3 Save Final Comparison Report

## Notes on the Original Outline

Two headings named models that do not exist in this project, so they were mapped onto the equivalents that are used here:

| Outline | Here | Why |
|---|---|---|
| `yolov8n-cls` | `yolo26n-cls` | the project uses YOLO26 throughout; the weights in `models/` are YOLO26 |
| `yolov8s-cls` / `yolov8m-cls` | `yolo26s-cls` / deeper CNN | same one step up the size ladder, per section 7.5 |

The 80/20 split in the original notebook also became 70/15/15, because section 4.7 and section 5 need a test set that is kept separate from validation.
