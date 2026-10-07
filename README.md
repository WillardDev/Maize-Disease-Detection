# Maize Leaf Disease Classification & Kenya Maize Advisory Chatbot

## Summary of the Project

Two projects in one repository:

1. **Image classification** — classify a maize leaf photo into **Blight**, **Common Rust**, **Gray Leaf Spot** or **Healthy**, comparing a pretrained YOLO26 classifier against a custom CNN built from scratch in Keras.
2. **Advisory chatbot (RAG)** — an English/Swahili retrieval-augmented chatbot that answers questions about maize farming in Kenya from a curated PDF knowledge base, integrated as a third tab in the same Streamlit app. Farmer profiles and full chat history are stored in **MySQL**.

A farmer can photograph a leaf, get an instant diagnosis, then continue the conversation for management advice — in English or Swahili. Conversations are saved so they can be revisited and analysed later.

## Problem Statement

Maize foliar diseases spread rapidly and can cut yields by 30–50%. The current diagnosis method is visual inspection by a human agronomist: slow, subjective, error-prone, and impossible to scale across large plantations. Many smallholder farmers have no access to a plant pathologist, so disease is caught only after irreversible damage. Even when a disease is identified, reliable agronomy advice — fertiliser, planting, storage, pest control — remains hard to access, especially in local languages.

## Project Objective

- Classify a maize leaf image into one of four classes so a farmer can photograph a leaf and get an instant diagnosis.
- Build and evaluate a free, locally-runnable RAG chatbot that provides grounded, source-cited answers on Kenyan maize farming in English and Swahili.

## Research Questions

**Image classification:**

1. How accurately can a pretrained YOLO26 classifier separate these four classes, and how much of that is transfer learning rather than training from scratch?
2. How close does a small custom CNN get, and is it small and fast enough to be worth deploying instead?
3. Which classes get confused with each other, and why — background clutter, lighting, or the visual overlap between Blight and Gray Leaf Spot?
4. How do model size, augmentation, and training time trade off against accuracy on a dataset this small?
5. Which pixels actually drive the final model's decision, and is it reading the lesion or the background?

**Advisory chatbot (RAG):**

6. How well does hybrid retrieval (BM25 + dense vectors) serve Kenyan maize farming queries compared with dense-only retrieval?
7. Can cross-lingual embeddings retrieve English corpus content from Swahili queries effectively?
8. Does the translation-pivot architecture (Swahili → English → Swahili) produce answers comparable in quality to native English answers?
9. How do chunk size, overlap and top-k affect retrieval accuracy and answer faithfulness?
10. How reliably do guardrails keep answers on-topic, grounded in sources, and free of stale market/subsidy claims?

## Data Sources

**Image classification dataset**

Source: [Kaggle — Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghouse/corn-or-maize-leaf-disease-dataset) — 4,186 usable images (4,188 files minus 2 cross-class duplicates), split 70/15/15 into train (3,664, classes levelled to 916 each), val (625) and test (625).

| Class | Images | Share |
|---|---|---|
| Common_Rust | 1,306 | 31.2% |
| Healthy | 1,162 | 27.8% |
| Blight | 1,146 | 27.4% |
| Gray_Leaf_Spot | 572 | 13.7% |
| **Total** | **4,186** | **100%** |

**RAG knowledge base** (`pdfs/`)

| Document | Content |
|---|---|
| `MAIZE GROWERS GUIDE.pdf` | Practical maize agronomy handbook — planting, fertiliser, field management |
| `National-Agriculture-Production-Report-2025.pdf` | National production statistics and sector context |
| `Planting_strategies_of_maize_farmers_in_Kenya_a_si.pdf` | Research article: planting strategies of Kenyan maize farmers |
| `Olwande_Smallholder_Maize_Efficiency_Kenya.pdf` | Study on smallholder maize efficiency in Kenya |
| `diversity in maize production environments and practices.pdf` | Maize production environments and on-farm practices |
| `DISEASE_INFO` (from `app.py`) | Seed knowledge for the four leaf-disease classes |

**MySQL database** (`maize_advisory`)

| Table | Key fields | Purpose |
|---|---|---|
| `farmers` | `id, full_name, county, preferred_lang, created_at` | Farmer profiles (no passwords — select or create a profile in the sidebar) |
| `chat_sessions` | `id, farmer_id, started_at, language` | One conversation thread |
| `chat_messages` | `id, session_id, farmer_id, role, message, language, sources, feedback, created_at` | Full chat history, retrieved citations, feedback |

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
│   ├── yolo26s-cls.pt                 # larger variant (not committed)
│   └── maize_disease_yolo26s.pt       # final trained model - TRACKED IN GIT, the app needs it
├── pdfs/                             # RAG knowledge base (source PDFs)
├── Maize_Disease_Detection_YOLO.ipynb # image classification pipeline (sections 0-11)
├── RAG_Advisory_Chatbot.ipynb         # RAG chatbot pipeline (10 sections)
├── rag_pipeline.py                    # chatbot inference module used by app.py
├── rag_data/                          # raw corpus + chunks.jsonl
├── rag_artifacts/                     # vector index, config, evaluation report
├── rag_logs/                          # query / feedback logs
├── app.py                             # single-file Streamlit app (detection, advisory chat, metrics)
├── .streamlit/config.toml             # headless server + dark theme for deployment
├── packages.txt                       # apt libs Streamlit Cloud needs for OpenCV
├── requirements.txt
├── README.md
└── model_comparison.md                # final classification report, written by section 11.3
```

## Notebooks Structure

**`Maize_Disease_Detection_YOLO.ipynb`**

- **Environment Setup & Configuration**
- **1. Data Preparation & Validation**
- **2. Create Dataset Folder Structure**
- **3. Model A: YOLO26n-cls Pipeline**
  - *Transfer learning: what it actually does here*
- **4. Model B: Custom CNN Pipeline**
- **5. Model Comparison**
- **6. Error Analysis (Best Model)**
- **7. Error Reduction (Best Model)**
- **8. Model Explainability (Grad-CAM)**
- **9. Inference & Visual Inspection (Final Model)**
- **10. Limitations & Intended Use**
- **11. Model Export & Deployment (Optional)**

**`RAG_Advisory_Chatbot.ipynb`**

- **1. Environment & Setup**
- **2. Data Ingestion & Chunking**
- **3. Embeddings & Vector Store**
- **4. Retrieval & Augmentation**
- **5. Evaluation**
- **6. Guardrails & Safety**
- **7. Optimization & Tuning**
- **8. Inference Pipeline**
- **9. Export**
- **10. Monitoring & Feedback**
- **11. Database & Chat History**