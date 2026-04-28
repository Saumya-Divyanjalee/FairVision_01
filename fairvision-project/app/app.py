"""
FairVision - Age Group Classification Demo (FINAL FIXED)
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ✅ PATH FIX (robust)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "m1_weighted_loss_best.pth")

st.set_page_config(
    page_title="FairVision - Age Group Classifier",
    page_icon="👁️",
    layout="centered"
)

AGE_NAMES = ["0-2", "3-9", "10-19", "20-29", "30-39",
             "40-49", "50-59", "60-69", "70+"]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ================= MODEL =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # ✅ FIX
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= LOAD MODEL =================
 
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        return None, device, "❌ File not found"

    try:
        model = FairVisionCNN(num_classes=9)
        state_dict = torch.load(MODEL_PATH, map_location=device)

        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)

        return model, device, None

    except Exception as e:
        return None, device, str(e)
# ================= IMAGE TRANSFORM =================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

# ================= PREDICT =================
def predict(model, image, device):
    img_tensor = get_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1).squeeze().cpu().numpy()

    top3_idx = probs.argsort()[::-1][:3]
    return [(AGE_NAMES[i], probs[i]) for i in top3_idx], probs

# ================= UI =================
st.title("👁️ FairVision")
st.subheader("CNN-Based Age Group Classification")

st.markdown("""
FairVision predicts **age group** from a face image using a CNN model.
""")

st.divider()

model, device, error = load_model()

# ================= ERROR HANDLING =================
if error:
    st.error(f"🚨 Model Load Error: {error}")
    st.code(MODEL_PATH)
else:
    st.success("✅ Model loaded successfully")

    uploaded_file = st.file_uploader(
        "📸 Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Predicting..."):
                results, all_probs = predict(model, image, device)

            st.markdown("### 🎯 Top Predictions")
            for i, (label, prob) in enumerate(results, 1):
                st.markdown(f"**#{i} — {label}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.2f}%")

        st.divider()

        import pandas as pd
        df = pd.DataFrame({
            "Age Group": AGE_NAMES,
            "Confidence": all_probs
        }).set_index("Age Group")

        st.bar_chart(df)

st.divider()

st.markdown("""
### ⚠️ Notes
- Model must match architecture
- `.pth` file must be correct
- Works for **educational/demo purposes**
""")