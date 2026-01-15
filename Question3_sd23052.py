import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import requests
import pandas as pd

# ------------------------------- #
# Page Setup #
# ------------------------------- #
st.set_page_config(
    page_title="Live Image Classification",
    layout="centered"
)

st.title("📷 Real-Time Image Classification Web App")
st.caption("Pretrained ResNet-18 Model | Computer Vision with OpenCV Output")

# ------------------------------- #
# Step 1: Library Setup #
# ------------------------------- #
device = torch.device("cpu")

# ------------------------------- #
# Step 2: Fetch ImageNet Class Labels #
# ------------------------------- #
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(IMAGENET_CLASSES_URL)
class_labels = response.text.splitlines()

# ------------------------------- #
# Step 3: Load Pretrained ResNet-18 #
# ------------------------------- #
cnn_model = models.resnet18(pretrained=True)
cnn_model.eval()
cnn_model.to(device)

# ------------------------------- #
# Step 4: Image Preprocessing Steps #
# ------------------------------- #
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Normalization for ImageNet pretrained models
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------- #
# Step 5: Capture Image from Webcam #
# ------------------------------- #
st.subheader("📸 Capture Image")
captured_img = st.camera_input("Take a photo with your webcam")

if captured_img is not None:
    img = Image.open(captured_img).convert("RGB")

    st.image(img, caption="Captured Image", use_column_width=True)

    # Preprocess the Image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # ------------------------------- #
    # Step 6: Model Prediction & Softmax Calculation #
    # ------------------------------- #
    with torch.no_grad():
        model_output = cnn_model(input_batch)

    softmax_probs = F.softmax(model_output[0], dim=0)

    # Get Top 5 Predictions
    top5_probs, top5_indices = torch.topk(softmax_probs, 5)

    # Prepare Result DataFrame
    prediction_data = pd.DataFrame({
        "Class Label": [class_labels[idx] for idx in top5_indices],
        "Probability": top5_probs.cpu().numpy()
    })

    # ------------------------------- #
    # Display Results #
    # ------------------------------- #
    st.subheader("🔍 Top-5 Predictions")
    st.dataframe(prediction_data, use_container_width=True)

    st.bar_chart(
        prediction_data.set_index("Class Label"),
        horizontal=True
    )

    st.success(
        f"Top Prediction: {prediction_data.iloc[0]['Class Label']} "
        f"({prediction_data.iloc[0]['Probability']:.2%})"
    )

    st.info(
        "Softmax probabilities represent the confidence of the predicted classes."
    )

