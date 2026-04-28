# FairVision 👁️
### CNN-Based Age Group Classification with Bias Detection & Mitigation

> IJSE | Certified AI & ML Engineer | Individual Assignment 2025/2026

---

## Project Overview

FairVision is a CNN-based age group classification system built on the **FairFace dataset**.
It detects and mitigates demographic bias across race and gender groups using two strategies:
1. **Class-Weighted Loss** (M1)
2. **Balanced Mini-Batches via WeightedRandomSampler** (M2)

---

## Project Structure

```
fairvision-project/
├── notebooks/
│   └── main_notebook.py       ← Full pipeline (convert to .ipynb)
├── app/
│   └── app.py                 ← Streamlit demo app
├── models/
│   ├── baseline_best.pth
│   ├── m1_weighted_loss_best.pth
│   └── m2_balanced_sampler_best.pth
├── outputs/
│   ├── plots/                 ← All generated charts
│   └── results/               ← CSV comparison tables
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook (train models + generate outputs)
jupyter notebook notebooks/main_notebook.ipynb

# 3. Run the Streamlit demo (after training)
cd app
streamlit run app.py
```

---

## Dataset

- **FairFace** (HuggingFaceM4/FairFace, config: 0.25)
- Train: 86,744 samples | Validation (Test): 10,954 samples
- Labels: age (9 classes), gender (2 classes), race (7 classes)

---

## Models

| Model | Strategy | Notes |
|---|---|---|
| Baseline | Standard CrossEntropyLoss | No bias correction |
| M1 | Class-Weighted Loss | Weights inversely proportional to class frequency |
| M2 | WeightedRandomSampler | Balanced batches per age class |

---

## Streamlit Demo

The demo allows users to:
- Upload a face image
- View top-3 predicted age groups with confidence scores
- See full probability distribution

**Deploy to Streamlit Cloud:**
1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo → select `app/app.py`
4. Deploy

---

## Academic Integrity

All code, architecture decisions, analysis, and conclusions are original work.
Libraries used: PyTorch, HuggingFace Datasets, scikit-learn, Streamlit.
