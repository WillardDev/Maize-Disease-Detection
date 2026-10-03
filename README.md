# Maize Leaf Disease Classification: YOLO26 vs Custom CNN

Classify a maize leaf image into one of four classes — **Blight**, **Common Rust**, **Gray Leaf Spot**, **Healthy** — and compare two models trained on the same split: a pretrained YOLO26 classifier and a custom CNN built from scratch in Keras.

## Problem Statement

Maize foliar diseases spread rapidly and can cut yields by 30–50%. The current diagnosis method is visual inspection by a human agronomist: slow, subjective, error-prone, and impossible to scale across large plantations. Many smallholder farmers have no access to a plant pathologist, so disease is caught only after irreversible damage.

**Objective:** classify a maize leaf image into one of four classes so a farmer can photograph a leaf and get an instant diagnosis.

## Research Questions

1. How accurately can a pretrained YOLO26 classifier separate these four classes, and how much of that is thanks to transfer learning rather than training from scratch?
2. How close does a small custom CNN get, and is it small and fast enough to be worth deploying instead?
3. Which classes get confused with each other, and why — background clutter, lighting, or the visual overlap between Blight and Gray Leaf Spot?
4. How do model size, augmentation, and training time trade off against accuracy on a dataset this small?
5. Which pixels actually drive the final model's decision, and is it reading the lesion or the background?

## Results

| Configuration | Val macro F1 | Val accuracy | Test accuracy |
|---|---|---|---|
| Model A — YOLO26n-cls (baseline) | 93.75% | 95.04% | 94.56% |
| Model B — Custom CNN | 88.65% | 88.96% | 89.44% |
| `exp_a` — stronger augmentation | 94.68% | 95.84% | — |
| **`exp_b` — YOLO26s-cls (final)** | **95.01%** | **96.00%** | **95.68%** |

The 96% target is met on validation (96.00%) but **not** on test (95.68%). The full report is written to `model_comparison.md`.

Selection is on **validation macro F1** only. Test is read once, in section 9.3, with the final model.

## Project Overview

A supervised image-classification pipeline with **two models** and a head-to-head comparison:

```
Data → Validation → 70/15/15 Split + Train Levelling
    → Model A: YOLO26n-cls (transfer learning)  ─┐
    → Model B: Custom CNN (Keras, from scratch)  ─┴→ Comparison → best model
    → Error Analysis → Error Reduction → Explainability → Inference → Export
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
| Healthy | 1,162 | 27.8% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 572 | 13.7% |
| **Total** | **4,186** | **100%** |

Real-world leaf photographs stored as `data/<Class_Name>/*.jpg`, where the folder name *is* the label. Files are indexed by **content hash** first, so the 2 byte-identical photographs that sat in two different class folders under different labels are dropped before anything else runs. 4,188 files become **4,186** usable images.

The split is **70 / 15 / 15**, and each part has one job:

- **train** — 3,664 images. The smaller classes are levelled up to 916 per class by **repetition**, done once in section 2.1 so both frameworks train on exactly the same images.
- **val** — 625 images, natural distribution. Every model-selection decision in the notebook is made here.
- **test** — 625 images, natural distribution. Read once, in 9.3, with the final model.

Levelling the *training* split does not create new photographs, so Gray Leaf Spot is still the hardest class underneath. That limitation is documented in section 10 rather than hidden.

**Known issues in the raw data:**

- Mixed file extensions (`.jpg`, `.JPG`, `.jpeg`) — several hundred files end in uppercase `.JPG`, which a lowercase `*.jpg` glob silently misses.
- Mixed colour modes: 4 images are RGBA and 1 is CMYK.
- Resolutions from 180×116 up to 5184×5184, though 3,852 images are exactly 256×256.
- Non-uniform backgrounds and lighting.
- **2 cross-class duplicates**: the same photograph filed under both Blight and Gray Leaf Spot, so one of the two labels is simply wrong. Both are dropped.

## Project Structure

```
maize_disease/
├── data/                              # source images, one folder per class
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── models/
│   ├── yolo26n-cls.pt                 # Model A baseline (pretrained, not committed)
│   ├── yolo26s-cls.pt                 # larger variant (section 7.3, not committed)
│   └── maize_disease_yolo26s.pt       # final trained model - TRACKED IN GIT, the app needs it
├── Maize_Disease_Detection_YOLO.ipynb # the full 0-11 pipeline
├── app.py                             # single-file Streamlit app (detection, info, metrics)
├── .streamlit/config.toml             # headless server + dark theme for deployment
├── packages.txt                       # apt libs Streamlit Cloud needs for OpenCV
├── slides_maize_disease.pptx          # 19-slide talk: every figure with its insight
├── requirements.txt
├── README.md
└── model_comparison.md                # final report, written by section 11.3
```

Generated during a run and safe to delete:

- `data_cls/` — the 70/15/15 split, created by section 2.1 and rebuilt every run
- `runs/classify/`, `runs/cnn/` — training logs, plots, confusion matrices, checkpoints
- `model_comparison.md` — the final report

## Requirements

```
pip install -r requirements.txt
```

That covers `ultralytics` (Model A), `tensorflow` (Model B), `opencv-python` (image reading), `scikit-learn` (confusion matrices and reports), `streamlit` (to run the app), `nbconvert` (to run the notebook end to end) and `python-pptx` (to rebuild the deck). `tensorflow` pulls in Keras. On Apple silicon it runs on the CPU, which is a large part of why Model B trains slower than Model A.

If you only want the app, `pip install streamlit ultralytics pillow` is enough — the notebook's training stack is not required.

`requirements.txt` pins the exact versions this project was verified against.

> **Note:** this project was developed against `tensorflow 2.22.0rc0` with `keras-nightly`, because no stable TensorFlow release supports Python 3.14. A stable Python (3.11–3.12) with a stable TensorFlow is recommended for reproducibility.

## Running the App

`app.py` is self-contained and dark-themed — take a photo with the camera or upload one, and the
app predicts the class and shows the confidence breakdown for every class.

```bash
source .venv/bin/activate
streamlit run app.py
```

To open it on a phone, bind it to your network and use the URL Streamlit prints:

```bash
streamlit run app.py --server.address 0.0.0.0
```

The app loads `models/maize_disease_yolo26s.pt` by default and caches it, so it only loads once
per session. It has two tabs: detection, and the measured model performance. Disease symptoms,
favourable conditions and what to do are shown for the class the model just predicted, not as a
catalogue of all four. It handles EXIF orientation, so photos taken on a phone are not rotated
incorrectly.

### Deploying: the weights must be in git

The 11 MB checkpoint is deliberately the one exception to the "keep binaries out of git" rule.
`.gitignore` ignores `*.pt`, but `models/maize_disease_yolo26s.pt` is negated back in, because
**a deploy from git cannot classify anything without it.** If the weights are absent, the sidebar
shows a red *Load failed* pill, **Run Detection** is disabled, and the app prints the exact
`git add -f` command to fix it.

```bash
git add .gitignore app.py .streamlit requirements.txt
git add -f models/maize_disease_yolo26s.pt   # or just `git add -A`, the negation covers it
git commit -m "Make Streamlit deployable: track the fine-tuned weights"
git push
```

Two further points about a hosted deployment:

- The app will **not** silently fall back to `yolo26n-cls.pt` if the fine-tuned weights are
  missing. That backbone has never seen maize disease, so switching to it would emit confident
  nonsense about ImageNet classes. It is only used if the checkbox asks for it explicitly.
- `requirements.txt` still carries the full notebook stack, including `tensorflow==2.22.0rc0`.
  That is not needed to run the app and is the most likely source of a slow or failed Cloud
  build. Splitting it into a lean `requirements.txt` for the app plus
  `requirements-notebook.txt` is the recommended next step.

`.streamlit/config.toml` sets `headless`, a 20 MB upload cap and the dark theme, so the page is
dark before the injected CSS is even parsed.

### `packages.txt`: the libGL fix

If the app reports

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

the host is missing GUI libraries. Ultralytics imports OpenCV, and `opencv-python` is a **GUI**
build linked against X and GL — which a headless container does not have. The app is fine; the
system libraries are missing.

Streamlit Community Cloud installs apt packages listed one-per-line in `packages.txt` at the
repository root, so this repo ships that file:

```
libgl1
libglib2.0-0
libxcb1
libsm6
libxext6
libxrender1
libgomp1
```

Which library is named in the error varies with the base image — `libxcb.so.1` instead of
`libGL.so.1` is the same fault — so list them all rather than reacting to one message.

**Alternative:** install `opencv-python-headless`, which has no GUI dependencies. It works, but
`ultralytics` hard-depends on `opencv-python`, so both distributions end up writing the same
`cv2` directory. The `packages.txt` route avoids that entirely.

This was verified by running `app.py`'s own `load_model()` inside `python:3.12-slim` with nothing
but `packages.txt` installed — it loads the weights and classifies correctly.

### Camera capture on a phone

The **Use the camera** option uses `st.camera_input`, so there is no separate file step in the
field. Browsers only expose a camera on a *secure context*, which has one consequence worth
knowing before you hand this to someone in a field:

| How you reach the app | Camera works? |
| --- | --- |
| `streamlit run app.py`, then open `http://localhost:8501` on the same machine | Yes |
| `http://<lan-ip>:8501` from a phone | No — not a secure context |
| Behind an HTTPS tunnel or reverse proxy | Yes |

For phone capture, put HTTPS in front of Streamlit, for example with a tunnel:

```bash
cloudflared tunnel --url http://localhost:8501
```

HTTPS has to terminate somewhere other than the Streamlit process, so this is deployment work the
script cannot do for you. Without it, phone users fall back to the uploader.

## Running the Notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Maize_Disease_Detection_YOLO.ipynb
```

Expect roughly 25–35 minutes, most of it in section 7. Use the `maize-disease` kernel so TensorFlow resolves from the virtualenv.

## Using the Trained Model

```python
from ultralytics import YOLO

model = YOLO("models/maize_disease_yolo26s.pt")
results = model.predict("data/Healthy/Corn_Health (1).jpg", verbose=False)

result = results[0]                      # predict() returns a list, even for one image
label = result.names[result.probs.top1]
print(label, f"{float(result.probs.top1conf):.1%}")   # -> Healthy 100.0%
```

The model expects images resized to **256×256** and the class order `Blight, Common_Rust, Gray_Leaf_Spot, Healthy`.

## Notebook Structure

- **Environment Setup & Configuration**
  - Install Dependencies
  - Import Libraries
  - Configuration
  - Set Random Seeds (reproducibility)
- **1. Data Preparation & Validation**
  - 1.1 Dataset Paths
  - 1.2 Dataset Validation (content-hash dedupe)
  - 1.3 Class Inspection
  - 1.4 Class Distribution Check
- **2. Create Dataset Folder Structure**
  - 2.1 Generate Folder Structure, Split & Level the Training Classes
  - 2.2 Verify Split (leak check)
- **3. Model A: YOLO26n-cls Pipeline**
  - 3.1 Load Pre-trained Classification Model
  - 3.2 Configure Training Arguments
  - 3.3 Start Training
  - 3.4 Visualize Training Results
  - 3.5 Run Validation
  - 3.6 Display Metrics
  - 3.7 Confusion Matrix
  - 3.8 YOLO Metrics
  - *Transfer learning: what it actually does here*
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
  - 5.7 Select Best Model *(validation macro F1)*
- **6. Error Analysis (Best Model)**
  - 6.1 Collect Misclassified Images
  - 6.2 Per-Class Error Rate
  - 6.3 Top Confusion Pairs
  - 6.4 Confidence Distribution of Errors
  - 6.5 Visualize Misclassified Samples
  - 6.6 Identify Error Patterns (lighting, blur, background, etc.)
- **7. Error Reduction (Best Model)**
  - 7.1 Data Augmentation (flip, colour jitter)
  - 7.2 Hyperparameter Tuning (lr, epochs, batch size, img size)
  - 7.3 Try Larger Variant (yolo26s-cls OR deeper CNN)
  - 7.4 Re-train & Compare Metrics
  - 7.5 Pick the Winner & Check the Target
- **8. Model Explainability (Grad-CAM)**
  - 8.1 Grad-CAM on the Final Model's Biggest Confusions
  - 8.2 Lesion or Background: How Much of the Frame Lights Up
- **9. Inference & Visual Inspection (Final Model)**
  - 9.1 Single Image Prediction
  - 9.2 Batch Prediction
  - 9.3 Manual QA
  - 9.4 Predict on a Single Custom Image
- **10. Limitations & Intended Use**
- **11. Model Export & Deployment (Optional)**
  - 11.1 Export Model
  - 11.2 Save Model Weights
  - 11.3 Save Final Comparison Report