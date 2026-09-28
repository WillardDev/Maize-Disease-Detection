# Maize Leaf Disease Detection using CNN

## Problem Definition

Maize (corn) is one of the most widely cultivated staple crops globally, and its yield is
heavily threatened by foliar diseases such as **Common Rust**, **Gray Leaf Spot** and
**Northern Leaf Blight**. These diseases are visually similar in their early stages, spread
rapidly through the field, and if detected late can cause severe yield losses of 30–50% in
susceptible hybrids.

The traditional method of disease diagnosis depends on **visual inspection by human
agronomists**, which is:
- **Subjective and error-prone** — early-stage symptoms are easily confused, leading to
  misdiagnosis and incorrect treatment.
- **Slow and not scalable** — expert inspection cannot cover large plantations in time,
  which is critical because these diseases are most damaging during the early growth stages.
- **Costly** — it requires trained personnel and repeated field visits.

In many farming regions, especially smallholders, there is simply no access to a plant
pathologist. The result is that diseases are identified after irreversible damage has already
occurred. This project addresses that gap by building an **automated, image-based disease
classification system** that can be used as a low-cost first-line screening tool.

---

## Project Objective

To design, train and evaluate a **Convolutional Neural Network (CNN)** using deep learning
that automatically classifies maize leaf images into one of four categories —
**Healthy, Common Rust, Gray Leaf Spot, and Blight** — with high accuracy, so that a farmer
or agronomist can capture a leaf image and receive an instant, reliable diagnosis.

Specific objectives:
1. Build a robust CNN classifier for 4-class maize leaf disease classification.
2. Apply appropriate data cleaning and augmentation to handle the real-world, noisy nature of
   field-captured leaf images.
3. Compare multiple CNN architectures and select the best-performing model using
   appropriate evaluation metrics.
4. Perform error analysis to understand failure cases and reduce classification errors.
5. Deploy the final model as a simple, usable prediction tool.

---

## Project Overview

This is a **supervised deep learning / image classification** project. The workflow is:

```
Data Collection → Data Cleaning → EDA → Train/Val/Test Split
     → Data Augmentation → CNN Model Training → Evaluation
     → Error Analysis → Error Reduction → Prediction / Deployment
```

Images are read from disk, converted to tensors and normalised to the range expected by the
neural network. A CNN is then trained to learn discriminative features directly from raw
pixels (no hand-crafted feature engineering is required), learning low-level features
(edges, colour blobs) in early layers and high-level disease patterns (lesion shape, texture)
in deeper layers. The trained model is evaluated on an unseen test set, its errors are
inspected, and corrective measures are applied.

The dataset is **imbalanced** (Gray Leaf Spot is roughly half the size of the other classes),
so class-weighted training, oversampling and macro-averaged metrics are used to avoid a model
that simply favours the majority classes.

---

## Research Questions

1. **Which CNN architecture (custom CNN, VGG16, ResNet50, InceptionV3, MobileNetV2, or a
   transfer-learned model) achieves the highest classification accuracy on the 4-class maize
   leaf disease dataset, and how do model size and training time trade off against accuracy?**

2. **How do data augmentation and class-balancing strategies (rotation, flipping, colour
   jitter, class weights) affect the model's generalisation and its performance on the
   minority class (Gray Leaf Spot)?**

3. **Which disease pairs are most frequently confused by the model, and what visual and
   feature-level factors (background clutter, lighting, lesion overlap between Common Rust and
   Blight) explain those errors — and which corrective measures reduce them?**

---

## Data Source

<https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset>

**Dataset details:**

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.7% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 574 | 13.7% |
| **Total** | **4,188** | **100%** |

- **Source type:** Real-world, unstructured image data (JPEG/JPG leaf photographs).
- **Class balance:** Moderately imbalanced — Gray Leaf Spot has ~43% fewer images than
  Common Rust. Accuracy alone can therefore be misleading, so macro-F1 and per-class recall
  are reported.
- **Known issues in the raw data:** inconsistent file extensions (`.jpg`, `.JPG`, `.jpeg`),
  varied image resolutions and aspect ratios, non-uniform backgrounds, varying lighting and
  colour balance, occasional duplicate/overlapping captures, and label noise where healthy
  leaves show minor natural blemishes that resemble early-stage disease.
- **Local location:** `data/<Class_Name>/*.jpg` (already downloaded into this repository).

---

## Tools

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Development | Jupyter Notebook (`Maize_Disease_Detection.ipynb`) |
| Data Handling | NumPy, pandas |
| Deep Learning | PyTorch, torchvision (CNNs, transfer learning, pretrained ImageNet weights) |
| Computer Vision | OpenCV (image loading, resizing, cleaning), Pillow |
| Visualisation | Matplotlib, Seaborn |
| Evaluation | scikit-learn (metrics, confusion matrix, classification report) |
| Configuration | Python `os` module for path handling, Google Drive (optional, if running on Colab) |

---

## Project Structure

```
maize_disease/
│
├── data/                          # Raw dataset (4 class folders)
│   ├── Blight/                   #   1,146 images - Northern Leaf Blight
│   ├── Common_Rust/              #   1,306 images - Common Rust
│   ├── Gray_Leaf_Spot/           #     574 images - Gray Leaf Spot
│   └── Healthy/                  #   1,162 images - Healthy leaves
│
├── Maize_Disease_Detection.ipynb  # Main notebook: full end-to-end pipeline
│
├── README.md                      # Project documentation (this file)
│
└── requirements.txt               # Python dependencies (to be added)
```

Each class directory contains leaf images, and the **folder name is the class label** — the
model learns to associate leaf visual patterns with the disease name. Labels are read
automatically using `ImageFolder`, so adding new images to the relevant folder is
automatically picked up by the pipeline.

---

## Notebook Structure

The single notebook `Maize_Disease_Detection.ipynb` is organised into the following sections,
in order:

### 1. Introduction & Problem Statement
Defines the agricultural problem — visual diagnosis of maize foliar diseases is slow,
subjective and not scalable — and states the objective: build a CNN that classifies a leaf
image into Healthy, Common Rust, Gray Leaf Spot or Blight. Also lists the research questions
this notebook will answer.

### 2. Importing Libraries
Imports and installs the required packages (PyTorch, torchvision, OpenCV, NumPy, pandas,
Matplotlib, Seaborn, scikit-learn) and configures the notebook.

### 3. Data Loading
**Why needed:** The dataset is on disk as a folder-per-class structure. Loading must convert
raw JPEG files into a form the model can consume, and the labels must be attached correctly
from the folder names.

- Reads the `data/` directory and builds a `DataFrame` with columns `image_path`, `label` and
  `class_name` by walking the class folders.
- Prints the class-wise image counts and the total, and plots a bar chart of the distribution
  to make the imbalance visible immediately.

### 4. Data Cleaning
**Why needed:** Raw web-scraped image datasets contain corrupted files, duplicate captures,
inconsistent resolutions and irrelevant images. Feeding these to the model wastes capacity,
causes silent failures mid-training, and inflates the error rate.

- **Corrupted/zero-byte image check:** every image is opened and verified; unreadable files are
  dropped and reported. (Note: `data/Blight/Corn_Blight (1).jpeg` and its `.jpg` counterpart
  look like near-duplicates, which this stage is designed to catch.)
- **Exact duplicate removal** using file hashes, and **near-duplicate detection** using
  perceptual hashing, so the same leaf photographed twice does not appear in both the training
  and test sets (which would otherwise inflate accuracy).
- **Mislabel review:** a contact sheet of random samples per class is printed for visual
  sanity-checking of labels.
- **Normalisation of format:** RGB conversion, consistent resizing to 224 × 224, and removal of
  corrupt/irrelevant (non-leaf) images identified during review.
- A summary of how many images were kept vs. removed is printed.

### 5. Exploratory Data Analysis (EDA)
**Why needed:** Before choosing a model, we must know the size of the dataset, how balanced
the classes are, and whether the images and labels actually look correct. EDA prevents
training a model on data that cannot support it.

- **Volume check:** total image count and images per class — confirms the dataset is large
  enough to train a CNN.
- **Class balance analysis:** count and percentage bar chart; percentage difference between
  majority and minority class, and a check on whether the imbalance warrants class weights.
- **Image dimension analysis:** distribution of height, width and aspect ratio, to decide the
  input resolution for the CNN.
- **Pixel statistics:** channel-wise mean and standard deviation of a sample, used to justify
  the choice of normalisation values.
- **Visual sample inspection:** a grid of random images from each of the 4 classes to
  sanity-check labels and image quality.
- **Class-wise colour analysis:** mean RGB values per class — healthy leaves are more
  uniformly green, while diseased leaves contain yellow/brown/grey lesion pixels. This confirms
  the classes are visually separable and gives insight into which classes may be confused.

### 6. Train / Validation / Test Split
**Why needed:** The model must be evaluated on data it has never seen, or the reported
accuracy is meaningless. CNN training is expensive, so a **dedicated validation set** is
needed to monitor performance and drive early stopping, rather than relying on cross-validation.

- **Stratified split** so all four classes keep the same proportions in train, validation and
  test — essential because the dataset is imbalanced.
- Split ratio **70% train / 15% validation / 15% test** (≈ 2,930 / 630 / 630 images).
- Class counts per split are printed and verified.
- **DataLoaders** are created with the appropriate `batch_size` and `num_workers` for
  efficient mini-batch training.

### 7. Data Augmentation
**Why needed:** 4,188 images is modest for a CNN, so the model will overfit. Augmentation
artually multiplies the effective dataset size and teaches the model invariances that match
real field conditions (leaves are photographed at arbitrary angles and orientations).

Applied **on the training set only** (never on validation/test, which would corrupt the
evaluation):
- Random resized crop (scale 0.8–1.0) to handle varied leaf sizes in frame
- Random horizontal flip and random rotation (± 15–20°)
- Random affine translation and slight zoom
- Colour jitter (brightness, contrast, saturation) to handle field lighting variation
- Normalisation with the ImageNet mean/std (required for transfer-learned backbones)

### 8. Model Building & Training
**Why needed:** A CNN must be defined and optimised to learn discriminative features from raw
pixels; the architecture determines the model's capacity and its bias toward image data.

- **Custom CNN baseline:** a Sequential / custom `nn.Module` with Conv2D → BatchNorm → ReLU →
  MaxPool blocks, then Global Average Pooling and a Dropout + Dense classifier head. Built
  first as a from-scratch reference point.
- **Transfer learning models** compared using ImageNet-pretrained backbones: VGG16,
  ResNet50, InceptionV3, MobileNetV2. The final classification layer is replaced with a
  `Linear(512, 4)` head (or the backbone is frozen and a new head attached, then optionally
  fine-tuned).
- **Optimiser:** Adam with a reduced learning rate for fine-tuning.
- **Loss function:** Cross-Entropy Loss, with `class_weights` computed from the training split
  to penalise errors on minority classes more heavily.
- **Training loop:** forward pass computes predictions, the loss is computed, backpropagation
  computes gradients, and the optimiser updates weights — repeated over many **epochs**. The
  loop records training and validation loss/accuracy per epoch.
- **Early stopping** with `ReduceLROnPlateau`, saving the best model checkpoint by validation
  loss.
- Learning curves for training vs. validation loss and accuracy are plotted after training.

### 9. Evaluation
**Why needed:** Accuracy alone is misleading on an imbalanced dataset and does not reveal
*which* class the model fails on. A set of complementary metrics is required to judge the model
fairly.

- **Test-set performance:** Accuracy, **Precision, Recall, F1-score, ROC-AUC**.
- **Macro-averaged** Precision / Recall / F1 (each class weighted equally) so the minority
  Gray Leaf Spot class is not hidden, alongside weighted-average and overall accuracy.
- **Per-class report** printed via `classification_report` to expose under-performing classes.
- **Confusion matrix** heatmap to visualise exactly which true classes are misclassified as
  which predicted classes.
- **ROC-AUC curve** (one-vs-rest) and **precision–recall curve** per class.
- Results also reported on the **validation set** to confirm the test result is not a fluke.

### 10. Error Analysis
**Why needed:** Metrics summarise performance but do not explain it. Looking at the actual
failure cases reveals whether the problem is overfitting, bad labels, confusing visual
similarity, or a weak architecture — each needing a different fix.

- **Overfitting / underfitting check:** compare training vs. validation loss curves — a large
  widening gap indicates overfitting; high error on both indicates underfitting.
- **Misclassified image inspection:** the highest-confidence wrong predictions and the
  lowest-confidence correct predictions are saved as a grid of actual vs. predicted labels.
  This reveals whether errors are concentrated in visually ambiguous, blurry, dark, cluttered
  or partially occluded images.
- **Class-wise error breakdown:** a bar chart of error rate per class to see which class
  contributes most to total error.
- **Per-sample error analysis:** the misclassified samples are grouped by *confidence* to test
  whether the model is confidently wrong (a labelling problem) or only uncertain at the
  decision boundary (a feature/architecture problem).
- Findings are written out as a short summary in markdown cells.

### 11. Error Reduction Strategies
**Why needed:** The concrete, justified response to the error analysis — turning identified
failure modes into measurable improvements.

Based on the error analysis, the following are applied and their effect on the metrics is
reported:

1. **Data augmentation tuning** — stronger geometric + colour augmentation, adding
   `RandomResizedCrop`, to improve robustness and reduce overfitting.
2. **Class imbalance handling** — class-weighted loss and/or oversampling of Gray Leaf Spot to
   improve minority-class recall.
3. **Transfer learning / better backbone** — if the custom CNN underperforms, adopt a stronger
   pretrained backbone (ResNet50 / InceptionV3).
4. **Fine-tuning strategy** — initial head-only training, followed by unfreezing later
   backbone layers at a lower learning rate to squeeze out more performance.
5. **Hyperparameter tuning** — learning rate, batch size, dropout rate, weight decay, using the
   validation set (never the test set) for selection.
6. **Advanced augmentation** — CutMix / MixUp, and background/crop-based augmentation to
   remove non-leaf background clutter.
7. **Threshold / calibration and ensembling** of several strong models by soft-voting to
   average out individual errors.
8. **Image quality filtering** — reject or down-weight very dark, low-resolution or
   blurred images at inference time.

### 12. Model Saving & Prediction
- The best model checkpoint is saved (`best_model.pth`) along with the class label mapping.
- A `predict(image_path)` function loads the saved model, preprocesses an arbitrary image the
  same way as training, and returns the predicted class with its confidence score.
- Demonstrated on sample images from the test set and on any custom image the user provides.

### 13. Conclusion & Future Work
Summary of the best model's performance, the key insights gained from error analysis, and the
practical relevance to farmers. Future directions: deployment as a mobile app or on-device
edge model, expansion to more maize diseases and other crops, and integration with real-time
camera capture in the field.

---

## Methodology Reference: Neural Networks for This Task

### 1.4 Neural Networks — Artificial Neural Networks / Deep Learning (Supervised — Classification)

**When to use:** Large datasets with complex, highly non-linear patterns — especially
unstructured data (images, audio, text) — where hand-crafted features are hard to design.
For this project, disease lesions in maize leaves are defined by subtle combinations of
colour, texture, shape and spatial arrangement of patterns. Writing rules or hand-engineering
features (e.g. "yellow if hue between X and Y") is brittle and would not generalise across
lighting, camera and cultivar differences. A CNN learns these features automatically from
the raw pixels.

**Nature of the dataset needed:** Large volume of data ideally; numeric tensors (images as
pixel arrays). The 4,188 image dataset is moderate in size, which is workable because
transfer learning with ImageNet-pretrained backbones lets us start from learned generic
features and only need enough data to adapt them — rather than learning visual features from
scratch on a small dataset. The dataset benefits enormously from scale, which is why
augmentation and transfer learning are central to the approach.

**Data cleaning needed & why:**
- *Impute or discard missing values* — unreadable or corrupt image files would otherwise
  raise an error during training or silently produce a corrupt batch; they are identified and
  removed.
- *Remove duplicates* — near-identical images split across train and test inflate the accuracy
  estimate and give a misleadingly optimistic result.
- *Scale/normalise inputs* (critical for stable gradient descent) — pixel values are scaled
  to [0, 1] and standardised with the ImageNet mean/std, keeping input magnitudes small so
  gradients stay well-conditioned and the network converges without diverging.
- *Standardise resolution* — all images resized to 224 × 224 so they form consistent batches.
- *Augment data* (for images) to increase the effective dataset size — random crops, flips,
  rotations and colour jitter generate new training examples, reducing overfitting and
  building invariance to the way leaves are actually photographed.

**EDA needed & why:**
- *Check data volume is sufficient* — 4,188 images across 4 classes is enough for transfer
  learning but too few to train a deep CNN reliably from scratch, which determines the
  modelling approach.
- *Class balance* — reveals the imbalance (Gray Leaf Spot at 13.7% vs. Common Rust at 31.2%),
  which drives the use of class weights, macro-averaged metrics and stratified splitting.
- *Sample a subset of inputs (e.g. images) to sanity-check labels and quality* — reveals
  mislabelled, blurry or cluttered images, and colour statistics per class confirm the classes
  are visually separable.

**Train/test split:** Train / validation / test split, often with a dedicated validation set
used during training for early stopping (not just CV, since training is expensive). We use a
stratified 70 / 15 / 15 split. The validation set drives early stopping, learning-rate
scheduling and all hyperparameter tuning; the test set is held out and touched **only once**,
at the very end, to report the final unbiased performance. Cross-validation is impractical
here because training several CNNs over multiple folds would be computationally expensive
for little added information.

**Model training:** The forward pass computes predictions from the input pixels; the loss
function measures the error; backpropagation computes the gradients of the loss with respect
to every weight; and an optimiser (e.g. **Adam**, or SGD with momentum) updates the weights
over many epochs. We tune the architecture (depth, number of filters, kernel size), the
learning rate, the batch size and regularisation (dropout, weight decay, early stopping,
batch normalisation).

**Evaluation metrics:**
- *Classification:* **Accuracy, Precision, Recall, F1-score, ROC-AUC**. Because of the class
  imbalance, macro-averaged F1 and per-class recall are the primary metrics, with accuracy
  reported alongside.
- Also track **train/validation loss curves** to detect overfitting during training.
- The **confusion matrix** is used to identify which specific disease pairs are confused.

**Error analysis & why:** Plot loss/accuracy curves for train vs. validation to catch
overfitting or underfitting; inspect misclassified examples to see if labels, augmentation, or
the architecture need adjustment. This distinguishes a model that is *confidently wrong*
(likely label noise, requiring better cleaning) from one that is *uncertain at the boundary*
(likely a feature/capacity problem, requiring more training data, stronger augmentation or a
better backbone).

**Pros:** Learns complex non-linear patterns automatically, state of the art on unstructured
data, highly flexible architecture choices, no manual feature engineering required, and
transfer learning makes high accuracy achievable even with a moderate dataset.

**Cons:** Needs large data and compute, many hyperparameters to tune, prone to overfitting on
small data, and low interpretability ("black box") — a farmer cannot see *why* a leaf was
classified a certain way, which limits trust and makes debugging harder.

**Real-world examples:** Image recognition, speech-to-text, machine translation, large
language models — and in agriculture specifically, plant disease apps such as PlantVillage-style
systems, weed detection in autonomous sprayers, and crop-yield forecasting from aerial imagery.
