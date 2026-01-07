import torch
import sys
import os
import argparse
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pilot_net import PilotNet
from utils.data_loader import CarlaDataset
from utils.preprocessing import get_transforms, denormalize_image
from evaluation.metrics import calculate_metrics
from xai.grad_cam import GradCamExplainer
from xai.lime_explainer import LimeExplainer

class MockCarlaDataset(torch.utils.data.Dataset):
    def __init__(self, length=10, transform=None):
        self.length = length
        self.transform = transform
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Generate random image
        image = Image.fromarray(np.random.randint(0, 255, (66, 200, 3), dtype=np.uint8))
        # Generate random controls
        controls = torch.tensor([
            np.random.uniform(-1, 1), # Steer
            np.random.uniform(0, 1),  # Throttle
            np.random.uniform(0, 1)   # Brake
        ], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, controls

def verify_pipeline():
    print("=== Starting Pipeline Verification ===")
    
    # 1. Data Loading
    print("\n[1] Verifying Data Loading...")
    try:
        # Use a small subset for verification to save time
        train_transforms = get_transforms(train=True)
        # Use MOCK Dataset to avoid downloading massive HF dataset
        full_dataset = MockCarlaDataset(length=10, transform=train_transforms)
        
        # subset_indices = range(10) # Just 10 samples
        # train_subset = Subset(full_dataset, subset_indices)
        train_loader = DataLoader(full_dataset, batch_size=2, shuffle=True)
        
        images, controls = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, Controls shape: {controls.shape}")
        print("Data Loading: OK")
    except Exception as e:
        print(f"Data Loading Failed: {e}")
        return

    # 2. Model Definition
    print("\n[2] Verifying Model Definition...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = PilotNet().to(device)
        # print(model) # Reduce output noise
        # Ensure images are on same device
        output = model(images.to(device))
        print(f"Model Output shape: {output.shape}")
        print("Model Definition: OK")
    except Exception as e:
        print(f"Model Definition Failed: {e}")
        return

    # 3. Training Loop (Micro-training)
    print("\n[3] Verifying Training Step...")
    try:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        # Move controls to device
        controls = controls.to(device)
        initial_loss = criterion(output, controls).item()
        print(f"Initial Loss: {initial_loss}")
        
        # Train for a few steps
        for i, (imgs, ctrls) in enumerate(train_loader):
            imgs, ctrls = imgs.to(device), ctrls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, ctrls)
            loss.backward()
            optimizer.step()
            print(f"Step {i}, Loss: {loss.item()}")
            
        print("Training Step: OK")
    except Exception as e:
        print(f"Training Failed: {e}")
        return

    # 4. Evaluation
    print("\n[4] Verifying Evaluation...")
    try:
        model.eval()
        with torch.no_grad():
            val_out = model(images.to(device))
            calculate_metrics(val_out.cpu().numpy(), controls.cpu().numpy())
        print("Evaluation: OK")
    except Exception as e:
        print(f"Evaluation Failed: {e}")
        return

    # 5. XAI Visualization (Grad-CAM)
    print("\n[5] Verifying XAI (Grad-CAM)...")
    try:
        # Pick the last convolutional layer. 
        # In PilotNet structure: conv1, conv2, conv3, conv4, conv5.
        target_layer = model.conv5
        explainer = GradCamExplainer(model, target_layer)
        
        # Explain the first image in the batch
        input_image = images[0].unsqueeze(0).to(device) # (1, 3, 66, 200)
        
        # Explain 'Steering' (index 0)
        heatmap = explainer.explain(input_image, target_index=0)
        print(f"Grad-CAM Heatmap shape: {heatmap.shape}, Max: {heatmap.max()}, Min: {heatmap.min()}")
        
        # Visualize
        # Denormalize input image for visualization overlay
        original_pil = denormalize_image(images[0])
        original_np = np.array(original_pil).astype(np.float32) / 255.0
        
        overlay = explainer.visualize(original_np, heatmap)
        
        # Save result
        output_path = "grad_cam_verification.png"
        import cv2
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM overlay saved to {output_path}")
        print("XAI (Grad-CAM): OK")
        
    except Exception as e:
        print(f"XAI (Grad-CAM) Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. XAI Visualization (LIME)
    print("\n[6] Verifying XAI (LIME)...")
    try:
        lime_exp = LimeExplainer(model)
        print("LIME Explainer initialized.")
        
        # LIME expects numpy (H, W, 3).
        # We use the denormalized image from above: original_np (float 0-1) or original_pil
        # Let's use the PIL image converted to numpy as per user request
        img_np = np.array(original_pil) # (66, 200, 3) uint8
        
        print("Generating LIME explanation (small samples for speed)...")
        # Reduce samples for verification speed
        lime_overlay = lime_exp.explain(img_np, target_index=0, num_samples=50) 
        
        print(f"LIME Overlay shape: {lime_overlay.shape}")
        
        # Save result
        output_path_lime = "lime_verification.png"
        # lime_overlay is likely float 0-1 (from skimage mark_boundaries) or similar.
        # mark_boundaries returns float [0, 1].
        
        # Convert to 0-255 uint8 for saving
        lime_save = (lime_overlay * 255).astype(np.uint8)
        
        cv2.imwrite(output_path_lime, cv2.cvtColor(lime_save, cv2.COLOR_RGB2BGR))
        print(f"LIME overlay saved to {output_path_lime}")
        print("XAI (LIME): OK")
        
    except Exception as e:
        print(f"XAI (LIME) Failed: {e}")
        import traceback
        traceback.print_exc()
        return


    print("\n=== Pipeline Verification Complete ===")

if __name__ == "__main__":
    verify_pipeline()
