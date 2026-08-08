# FairVision - CNN-Based Age Group Classification with Bias Mitigation

> Detecting and Mitigating Demographic Bias in a Face Analytics System Using FairFace

**IJSE | Certified AI & ML Engineer | Individual Assignment | 2025/2026**

---

## Overview

FairVision is a custom CNN-based age group classification system trained on the [FairFace dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace). The project goes beyond standard image classification by auditing the model for demographic bias across race and gender groups, and applying practical mitigation strategies to improve fairness.

---

## Demo Application

**Live URL:** https://fairvision01-sihb8rgzjrvxtw8mjuxfmj.streamlit.app

Upload a face image and receive:
- Top 3 predicted age groups with confidence scores
- Full probability distribution chart
- Model details and responsible use guidelines

---

## Project Structure

```
FairVision_01/
└── fairvision-project/
    ├── app/
    │   ├── app.py                    # Streamlit demo application
    │   ├── models/
    │   │   └── m1_weighted_loss_best.pth
    │   └── .streamlit/
    │       └── config.toml
    ├── notebooks/
    │   └── main.ipynb                # Main executed notebook (submit)
    ├── models/
    │   ├── baseline_best.pth
    │   ├── m1_weighted_loss_best.pth
    │   └── m2_balanced_sampler_best.pth
    ├── outputs/
    │   ├── plots/                    # All EDA and evaluation charts
    │   └── results/
    │       └── comparison_table.csv
    ├── src/
    │   ├── data/dataset.py
    │   ├── fairness/evaluator.py
    │   ├── models/cnn_model.py
    │   └── training/trainer.py
    └── requirements.txt
```

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | HuggingFace — HuggingFaceM4/FairFace |
| Config | 0.25 (tightly cropped, 224x224) |
| Train split | 86,744 images |
| Validation split | 10,954 images |
| Age classes | 9 groups (0-2 to 70+) |
| Race classes | 7 groups |
| Gender classes | 2 (Male, Female) |

```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")
```

---

## CNN Architecture

Custom 5-block CNN designed from scratch using PyTorch. No pretrained models or transfer learning used.

```
Input (3, 224, 224)
    ↓
ConvBlock 1: Conv2d(3→32)   + BN + ReLU + MaxPool → (32, 112, 112)
ConvBlock 2: Conv2d(32→64)  + BN + ReLU + MaxPool → (64, 56, 56)
ConvBlock 3: Conv2d(64→128) + BN + ReLU + MaxPool → (128, 28, 28)
ConvBlock 4: Conv2d(128→256)+ BN + ReLU + MaxPool → (256, 14, 14)
ConvBlock 5: Conv2d(256→256)+ BN + ReLU + MaxPool → (256, 7, 7)
    ↓
Flatten → 12,544
Dropout(0.5) → FC(1024) → ReLU
Dropout(0.3) → FC(512)  → ReLU
FC(9) → Output (9 age groups)
```

**Total parameters: 14,354,729**

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | StepLR (step=3, gamma=0.5) |
| Epochs | 3 (CPU constraint) |
| Batch size | 64 |
| Training samples | 10,000 (stratified subset) |
| Device | CPU (Google Colab) |

---

## Results

### Overall Performance

| Model | Accuracy | Macro F1 | Race Gap |
|-------|----------|----------|----------|
| Baseline | 0.3083 | 0.0753 | 0.1617 |
| M1: Class-Weighted Loss | 0.1013 | 0.0667 | **0.0396** |
| M2: Balanced Mini-Batches | 0.2218 | **0.1068** | 0.1027 |

### Fairness Audit — Baseline Race Accuracy

| Race Group | Accuracy |
|------------|----------|
| White | 0.4252 (BEST) |
| East Asian | 0.2635 (WORST) |
| Race Gap | 0.1617 |

### Fairness Audit — Baseline Gender Accuracy

| Gender | Accuracy |
|--------|----------|
| Male | 0.2612 |
| Female | 0.3611 |

---

## Bias Mitigation Strategies

### M1 — Class-Weighted Loss
- **Why:** Age class imbalance causes model to favour majority classes (20-29: 29.51%, 30-39: 22.19%)
- **How:** `weight[i] = 1/count[i]`, normalised to sum=9, passed to `CrossEntropyLoss(weight=w_tensor)`
- **Result:** Race gap reduced by **75%** (0.1617 → 0.0396)

### M2 — Balanced Mini-Batches
- **Why:** Complementary approach addressing data sampling rather than loss function
- **How:** `WeightedRandomSampler` ensures balanced age class representation per batch
- **Result:** Best Macro F1 (0.1068), moderate race gap reduction (0.1617 → 0.1027)

---

## Key EDA Findings

| Age Group | Count | % of Train | Risk |
|-----------|-------|------------|------|
| 0-2 | 1,792 | 2.07% | HIGH |
| 20-29 | 25,598 | 29.51% | Low (majority) |
| 70+ | 842 | 0.97% | HIGH |

- **Primary bias risk:** Age class imbalance — 0-2 and 70+ severely underrepresented
- Race distribution relatively balanced across 7 groups
- Gender distribution approximately balanced (52% Male, 48% Female)

---

## Final Recommendation

**Recommended model: M2 — Balanced Mini-Batches**

Provides the best balance between fairness and accuracy:
- Highest Macro F1 (0.1068)
- Reasonable accuracy (22.18%)
- Meaningful race gap reduction (0.1027)

**Deployment verdict: CONDITIONALLY ACCEPTABLE** for low-stakes controlled environments only.

Required conditions:
- Human oversight for all consequential decisions
- Regular bias re-audits as new data arrives
- Transparent disclosure of demographic performance gaps

**NOT suitable for:** Legal age verification, medical decisions, surveillance, or high-stakes automated decisions.

---

## Run Locally

### Prerequisites
```bash
pip install torch torchvision streamlit pillow numpy pandas
```

### Run Demo App
```bash
cd fairvision-project/app
streamlit run app.py
```

### Run Notebook
Open `notebooks/main.ipynb` in Google Colab or Jupyter Notebook.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision |
| Data | HuggingFace Datasets, pandas, numpy |
| Evaluation | scikit-learn |
| Visualization | matplotlib, seaborn |
| Demo | Streamlit |
| Version Control | Git, GitHub |
| Training | Google Colab |

---

## Academic Information

| Detail | Value |
|--------|-------|
| Student | Saumya Divyanjalee |
| Programme | Certified AI & ML Engineer (CAME) |
| Institute | IJSE — Institute of Software Engineering |
| Lecturer | Dasun Athukorala |
| Academic Year | 2025/2026 |
| Assignment Weight | 50% |

---

## References

- Karkkainen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age. WACV 2021.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
- HuggingFace. (2024). FairFace Dataset. https://huggingface.co/datasets/HuggingFaceM4/FairFace

---

*Developed as part of the CAME Individual Assignment — IJSE | 2025/2026*
