import streamlit as st
import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pilot_net import PilotNet
from xai.explainer import XAIController
from utils.preprocessing import preprocess_image, denormalize_image

st.set_page_config(page_title="XAI Autonomous Driver", layout="wide")

st.title("Explainable AI for Autonomous Driving")

# Sidebar
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", "models/pilot_net_best.pth")
xai_method = st.sidebar.selectbox("XAI Method", ["grad_cam", "lime", "shap"])

# Load Model
@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model, device
    else:
        return None, device

model, device = load_model(model_path)

if model is None:
    st.error(f"Model not found at {model_path}. Please train the model first.")
else:
    st.success("Model loaded successfully.")
    
    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict
        if st.button("Analyze"):
            img_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                steer, throttle, brake = output.cpu().numpy()[0]
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Steering", f"{steer:.2f}")
            col2.metric("Throttle", f"{throttle:.2f}")
            col3.metric("Brake", f"{brake:.2f}")
            
            # Explain
            st.subheader(f"Explanation ({xai_method})")
            
            # Init XAI
            # Background for SHAP
            background = torch.zeros((10, 3, 66, 200)).to(device)
            # Target layer for GradCAM
            target_layer = model.conv5
            
            xai = XAIController(model, background_data=background, target_layer=target_layer)
            
            with st.spinner("Generating explanation..."):
                heatmap = xai.explain(xai_method, img_tensor, input_image_np=np.array(image), target_index=0)
                
            if heatmap is not None:
                # Visualize
                # We need to overlay metrics
                # Resize heatmap to original image size for display? 
                # Or just use the 66x200 normalized/denormalized.
                
                # heatmap is (66, 200) usually or (H, W) from gradcam.
                # Let's resize heatmap to image size
                heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
                
                # Normalize heatmap for display (0-255)
                heatmap_vis = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                original_np = np.array(image)
                overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
                
                st.image(overlay, caption=f"{xai_method} Heatmap", use_column_width=True)
                
            else:
                st.error("Failed to generate heatmap.")
