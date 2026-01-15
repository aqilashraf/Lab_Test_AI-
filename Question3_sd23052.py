import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import pandas as pd
from typing import List

# =========================
# 1) Page Configuration
# =========================
st.set_page_config(
    page_title="VisionAI | Image Classifier",
    page_icon="📸",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 2) Optimized Classifier Class
# =========================
class ImageNetClassifier:
    def __init__(self, device: torch.device):
        self.device = device
        self.labels = self._load_labels()
        # Use st.cache_resource inside the app logic or call it externally
        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_labels(self) -> List[str]:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        return response.text.splitlines()

    def _load_model(self):
        # Using ResNet18 for speed/demo purposes
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        return model.to(self.device)

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 5) -> pd.DataFrame:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        
        probabilities = F.softmax(logits[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, top_k)

        return pd.DataFrame({
            "Class Label": [self.labels[i] for i in top_idx],
            "Probability": top_prob.cpu().numpy()
        })

# Initialize System with Caching to prevent re-loading weights
@st.cache_resource
def get_classifier():
    return ImageNetClassifier(torch.device("cpu"))

classifier = get_classifier()

# =========================
# 3) Sidebar & Settings
# =========================
with st.sidebar:
    st.title("⚙️ System Settings")
    st.info("ResNet-18 Model Loaded")
    
    top_k_slider = st.slider("Number of predictions to show", 3, 10, 5)
    
    st.divider()
    st.markdown("### 🧬 Inference Details")
    st.write(f"**Device:** CPU")
    st.write(f"**Backend:** PyTorch {torch.__version__}")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()

# =========================
# 4) Main Dashboard Layout
# =========================
st.title("📷 Real-Time Vision Intelligence")
st.markdown("---")

col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.subheader("📹 Optical Input")
    captured_image = st.camera_input("Scanner active...")
    
    if not captured_image:
        st.info("Waiting for camera input to begin classification.")
        # Placeholder image logic if needed
    
with col_right:
    st.subheader("📊 Analytical Output")
    
    if captured_image:
        with st.spinner("Processing through Neural Network..."):
            image = Image.open(captured_image).convert("RGB")
            results = classifier.predict(image, top_k=top_k_slider)
            
            # Key Metric
            top_class = results.iloc[0]['Class Label']
            top_conf = results.iloc[0]['Probability']
            
            st.metric(label="Primary Classification", value=top_class, delta=f"{top_conf:.2%} Confidence")
            
            # Visualizing Probabilities
            st.markdown("**Confidence Distribution**")
            # We use a progress bar for the top result to make it visual
            st.progress(float(top_conf))
            
            # Chart and Data
            tab_chart, tab_data = st.tabs(["📈 Confidence Chart", "📋 Raw Data"])
            
            with tab_chart:
                chart_data = results.set_index("Class Label")
                st.bar_chart(chart_data, color="#007bff", horizontal=True)
            
            with tab_data:
                st.dataframe(results, use_container_width=True, hide_index=True)
                
            st.success(f"Successfully identified as **{top_class}**")
    else:
        st.write("No data to display. Please capture an image to the left.")

# =========================
# 5) Education/Footnote
# =========================
with st.expander("ℹ️ How this works"):
    st.write("""
        1. **Pre-processing:** The image is resized to 224x224 pixels and normalized using ImageNet mean/standard deviation.
        2. **Forward Pass:** The pixels are fed through 18 layers of the ResNet architecture.
        3. **Softmax Layer:** The final output is converted into probabilities that sum to 100%.
    """)
