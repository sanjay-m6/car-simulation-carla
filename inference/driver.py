import torch
import cv2
import time
import numpy as np
import os
import sys

# Append project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from carla_interface.client import CarlaClient
from models.pilot_net import PilotNet
from xai.explainer import XAIController
from visualization.hud import HUD
from utils.preprocessing import preprocess_image

class Driver:
    def __init__(self, model_path, xai_method='grad_cam', target_layer_name='conv5'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = PilotNet().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: Model path not found, using random initialization.")
        self.model.eval()
        
        # XAI
        # For SHAP/LIME we might need background data. 
        # Using dummy background for now if needed.
        background = torch.zeros((10, 3, 66, 200)).to(self.device)
        
        # Determine target layer for Grad-CAM
        target_layer = None
        if xai_method == 'grad_cam':
            # Map string name to layer
            if hasattr(self.model, target_layer_name):
                target_layer = getattr(self.model, target_layer_name)
            else:
                target_layer = self.model.conv5 # Default
                
        self.xai = XAIController(self.model, background_data=background, target_layer=target_layer)
        self.xai_method = xai_method
        
        # Client & HUD
        self.client = CarlaClient()
        self.hud = HUD()
        
        # Safety Score State
        self.safety_score = 100.0
        self.last_steering = 0.0
        
    def run(self):
        try:
            self.client.setup_vehicle()
            self.client.setup_camera()
            
            # Warmup
            time.sleep(2)
            
            while True:
                start_time = time.time()
                
                # Get Data
                frame, frame_id = self.client.get_data()
                if frame is None:
                    continue
                    
                # Preprocess for model
                # Frame is BGR numpy
                img_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess_image(img_pil).to(self.device)
                
                # Predict
                with torch.no_grad():
                    output = self.model(input_tensor)
                    # output: [[steering, throttle, brake]]
                    controls = output.cpu().numpy()[0]
                    steering, throttle, brake = controls
                    
                # Apply Control
                self.client.apply_control(steering, throttle, brake)
                
                # Calculate Safety Score
                # Penalize jerky steering (derivative)
                steering_jerk = abs(steering - self.last_steering)
                penalty = 0.0
                if steering_jerk > 0.1: # Threshold for jerk
                    penalty += 2.0
                if brake > 0.8: # Hard braking
                    penalty += 0.5
                
                self.safety_score -= penalty
                self.safety_score += 0.05 # Slow recovery
                self.safety_score = float(np.clip(self.safety_score, 0, 100))
                self.last_steering = steering

                # Explanation
                heatmap = None
                if self.xai_method:
                    # Explain Steering (index 0) usually
                    heatmap = self.xai.explain(self.xai_method, input_tensor, input_image_np=frame, target_index=0)
                
                # Visualize
                # Overlay heatmap if exists
                explained_frame = frame.copy()
                if heatmap is not None and self.xai_method == 'grad_cam':
                     # Grad-CAM returns normalized heatmap.
                     # Visualize handles overlay.
                     # We need access to helper or implement here.
                     # Let's import visualize or implement simple one.
                     # Re-use the one in GradCamExplainer or copy logic?
                     # Better to have utility.
                     # For now, let's use the XAI Controller or the explainer instance if accessible.
                     # Actually explain returns heatmap.
                     
                     if self.xai_method == 'grad_cam':
                        # Ensure frame is float32 for visualization
                        frame_float = frame.astype(np.float32) / 255.0
                        explained_frame = self.xai.grad_cam.visualize(frame_float, heatmap)
                        explained_frame = (explained_frame * 255).astype(np.uint8)
                     elif self.xai_method == 'lime':
                        # LIME explanation returns image with boundaries
                        explained_frame = (heatmap * 255).astype(np.uint8)
                     elif self.xai_method == 'shap':
                        # Basic heatmap overlay
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                        explained_frame = cv2.addWeighted(frame.astype(np.float32), 0.6, heatmap_colored.astype(np.float32), 0.4, 0).astype(np.uint8)

                fps = 1.0 / (time.time() - start_time)
                hud_img = self.hud.render(frame, explained_frame, (steering, throttle, brake), fps, self.xai_method, self.safety_score)
                
                cv2.imshow("Explainable Autonomous Driver", hud_img)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.xai_method = 'grad_cam'
                elif key == ord('2'):
                    self.xai_method = 'lime'
                elif key == ord('3'):
                    self.xai_method = 'shap'
                elif key == ord('0'):
                    self.xai_method = None
                    
        finally:
            self.client.cleanup()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    driver = Driver("models/pilot_net_best.pth")
    driver.run()
