"""
FairVision - Age Group Classification Demo
CNN-Based Bias Detection and Mitigation System
SAUMYA DIVYANJALEE | IJSE | CAME | 2025/2026
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

 
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

 
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
# Try app/models/ first, then ../models/
_path1 = os.path.join(APP_DIR, "models", "m1_weighted_loss_best.pth")
_path2 = os.path.join(APP_DIR, "..", "models", "m1_weighted_loss_best.pth")
MODEL_PATH = _path1 if os.path.exists(_path1) else _path2

st.set_page_config(
    page_title="FairVision | Age Group Classifier",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1f77b4, #0d47a1); padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem; }
    .main-header h1 { color: white; font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0; }
    .metric-card { background: #1e2130; border: 1px solid #2d3250; border-radius: 8px; padding: 1.2rem; text-align: center; }
    .metric-card h3 { color: #1f77b4; font-size: 1.8rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #aaa; font-size: 0.82rem; margin: 0.3rem 0 0 0; }
    .pred-card { background: #1e2130; border-left: 4px solid #1f77b4; border-radius: 6px; padding: 0.75rem 1.2rem; margin: 0.4rem 0; }
    .pred-card.rank1 { border-left-color: #d4a017; }
    .pred-card.rank2 { border-left-color: #a8a8a8; }
    .pred-card.rank3 { border-left-color: #b87333; }
    .upload-section { background: #1e2130; border: 2px dashed #1f77b4; border-radius: 10px; padding: 3rem; text-align: center; margin-top: 1rem; }
    .upload-section h3 { color: #1f77b4; margin-bottom: 0.5rem; }
    .upload-section p  { color: #888; margin: 0; }
    .result-highlight { border-radius: 8px; padding: 1rem; text-align: center; margin-top: 1rem; }
    .footer { text-align: center; color: #555; padding: 1.5rem 0 0.5rem 0; font-size: 0.82rem; border-top: 1px solid #2d3250; margin-top: 2rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

AGE_NAMES = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
AGE_COLORS = {
    "0-2":"#e74c3c","3-9":"#e67e22","10-19":"#f39c12",
    "20-29":"#27ae60","30-39":"#16a085","40-49":"#2980b9",
    "50-59":"#2c3e50","60-69":"#8e44ad","70+":"#c0392b"
}
RANK_LABELS  = ["#1","#2","#3"]
RANK_CLASSES = ["rank1","rank2","rank3"]

if TORCH_OK:
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, pool=True):
            super().__init__()
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            if pool: layers.append(nn.MaxPool2d(2,2))
            self.block = nn.Sequential(*layers)
        def forward(self, x): return self.block(x)

    class FairVisionCNN(nn.Module):
        def __init__(self, num_classes=9):
            super().__init__()
            self.features = nn.Sequential(ConvBlock(3,32),ConvBlock(32,64),ConvBlock(64,128),ConvBlock(128,256),ConvBlock(256,256))
            self.classifier = nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(256*7*7,1024),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,num_classes))
        def forward(self, x): return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    if not TORCH_OK:
        return None, "cpu", "PyTorch not installed"
    device = torch.device("cpu")
    if not os.path.exists(MODEL_PATH):
        return None, device, f"Model not found: {MODEL_PATH}"
    try:
        model = FairVisionCNN(num_classes=9)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device, None
    except Exception as e:
        return None, "cpu", str(e)

def get_transform():
    return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=MEAN,std=STD)])

def predict(model, image, device):
    img_t = get_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img_t),dim=1).squeeze().cpu().numpy()
    top3 = probs.argsort()[::-1][:3]
    return [(AGE_NAMES[i],float(probs[i])) for i in top3], probs

# Sidebar
with st.sidebar:
    st.markdown("## FairVision")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("A bias-aware CNN trained on the [FairFace dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace) for age group classification with demographic fairness auditing.")
    st.markdown("---")
    st.markdown("### Age Groups")
    for age in AGE_NAMES:
        color = AGE_COLORS.get(age,"#888")
        st.markdown(f'<span style="color:{color};font-weight:bold;">&#9632;</span>&nbsp;{age}',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Model Details")
    st.markdown("| Detail | Value |\n|--------|-------|\n| Type | Custom CNN |\n| Framework | PyTorch |\n| Dataset | FairFace 0.25 |\n| Output | 9 Age Groups |\n| Bias Fix | Class-Weighted Loss |\n| Input | 224x224 RGB |")
    st.markdown("---")
    st.warning("For educational and research purposes only.")
    st.caption("IJSE | CAME | 2025/2026")

# Header
st.markdown('<div class="main-header"><h1>FairVision</h1><p>CNN-Based Age Group Classification with Bias Detection and Mitigation</p></div>',unsafe_allow_html=True)

model, device, error = load_model()
status_text  = "Ready" if not error else "Error"
status_color = "#27ae60" if not error else "#e74c3c"

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown('<div class="metric-card"><h3>9</h3><p>Age Groups</p></div>',unsafe_allow_html=True)
with c2: st.markdown('<div class="metric-card"><h3>86K</h3><p>Training Images</p></div>',unsafe_allow_html=True)
with c3: st.markdown('<div class="metric-card"><h3>7</h3><p>Race Groups Audited</p></div>',unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3 style="color:{status_color};font-size:1.1rem;">{status_text}</h3><p>Model Status</p></div>',unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

if error:
    st.error(f"Model Error: {error}")
    st.info(f"Model path checked: {MODEL_PATH}")
    st.stop()

st.markdown("## Upload a Face Image")
uploaded_file = st.file_uploader("Select a JPG or PNG face image",type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("---")
    st.markdown("## Analysis Results")
    left_col,right_col = st.columns([1,1])

    with left_col:
        st.markdown("### Uploaded Image")
        st.image(image,use_column_width=True)
        w,h = image.size
        st.caption(f"Dimensions: {w} x {h} px  |  Format: {uploaded_file.type}")

    with right_col:
        st.markdown("### Top 3 Predictions")
        with st.spinner("Analyzing image..."):
            results,all_probs = predict(model,image,device)
        for i,(label,prob) in enumerate(results):
            clr = AGE_COLORS.get(label,"#1f77b4")
            pct = prob*100
            st.markdown(f'<div class="pred-card {RANK_CLASSES[i]}"><div style="display:flex;justify-content:space-between;align-items:center;"><span style="color:white;font-size:1rem;">{RANK_LABELS[i]} &nbsp; Age Group: <strong style="color:{clr}">{label}</strong></span><span style="color:{clr};font-size:1.2rem;font-weight:bold;">{pct:.1f}%</span></div></div>',unsafe_allow_html=True)
            st.progress(float(prob))
            st.markdown("")
        top_label,top_prob = results[0]
        top_clr = AGE_COLORS.get(top_label,"#1f77b4")
        st.markdown(f'<div class="result-highlight" style="background:{top_clr}18;border:1px solid {top_clr};"><p style="color:#aaa;margin:0;font-size:0.85rem;">Most Likely Age Group</p><h2 style="color:{top_clr};margin:0.3rem 0;">{top_label}</h2><p style="color:#aaa;margin:0;">Confidence: {top_prob*100:.1f}%</p></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Full Probability Distribution")
    prob_df = pd.DataFrame({"Age Group":AGE_NAMES,"Confidence (%)":[round(p*100,2) for p in all_probs]}).set_index("Age Group")
    st.bar_chart(prob_df,color="#1f77b4")
    with st.expander("View Raw Prediction Data"):
        st.dataframe(pd.DataFrame({"Age Group":AGE_NAMES,"Confidence":[f"{p*100:.2f}%" for p in all_probs],"Raw Score":[f"{p:.6f}" for p in all_probs]}),use_container_width=True)
else:
    st.markdown('<div class="upload-section"><h3>Upload an Image to Begin</h3><p>Supported: JPG, JPEG, PNG<br>Use a clear front-facing face photo.</p></div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown("## Project Overview")
tab1,tab2,tab3 = st.tabs(["Model Architecture","Fairness and Bias","Limitations"])
with tab1:
    st.markdown("### Custom CNN (From Scratch)\n\nEach ConvBlock: Conv2d → BatchNorm2d → ReLU → MaxPool2d\n\n| Block | Input | Output | Size |\n|-------|-------|--------|------|\n| 1 | 3 RGB | 32 | 112x112 |\n| 2 | 32 | 64 | 56x56 |\n| 3 | 64 | 128 | 28x28 |\n| 4 | 128 | 256 | 14x14 |\n| 5 | 256 | 256 | 7x7 |\n\nClassifier: Flatten → Dropout(0.5) → FC(1024) → FC(512) → FC(9)\n\nOptimizer: Adam | Loss: CrossEntropyLoss + class weights | Scheduler: StepLR")
with tab2:
    st.markdown("### Bias Detection and Mitigation\n\n**Audit:** Race (7 groups) + Gender (Male/Female)\n\n**M1 — Class-Weighted Loss (Active):** Rare age classes get higher penalty weights.\n\n**M2 — Balanced Mini-Batches:** WeightedRandomSampler for balanced batches.")
with tab3:
    st.markdown("### Limitations\n\n**NOT for:** Legal age verification, medical decisions, surveillance.\n\n**OK for:** Research, education, analytics with human oversight.\n\n**Required:** Human review | Regular audits | Transparent disclosure.")

st.markdown('<div class="footer">FairVision — CNN-Based Age Group Classification with Bias Mitigation<br>CAME Individual Assignment | IJSE | 2025/2026<br><span style="color:#1f77b4;">Dataset: FairFace | Framework: PyTorch | UI: Streamlit</span></div>',unsafe_allow_html=True)