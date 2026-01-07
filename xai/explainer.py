from .grad_cam import GradCamExplainer
from .lime_explainer import LimeExplainer
from .shap_explainer import ShapExplainer
import torch

class XAIController:
    def __init__(self, model, background_data=None, target_layer=None):
        self.model = model
        
        # Initialize explainers
        self.grad_cam = GradCamExplainer(model, target_layer) if target_layer else None
        self.lime = LimeExplainer(model)
        
        if background_data is not None:
            self.shap = ShapExplainer(model, background_data)
        else:
            self.shap = None
            
    def explain(self, method, input_tensor, input_image_np=None, target_index=0):
        """
        method: 'grad_cam', 'lime', 'shap'
        """
        if method == 'grad_cam':
            if self.grad_cam:
                return self.grad_cam.explain(input_tensor, target_index)
            else:
                print("Grad-CAM not initialized (missing target_layer)")
                return None
        elif method == 'lime':
            if input_image_np is not None:
                return self.lime.explain(input_image_np, target_index)
            else:
                print("LIME requires input_image_np")
                return None
        elif method == 'shap':
            if self.shap:
                return self.shap.explain(input_tensor, target_index)
            else:
                print("SHAP not initialized (missing background_data)")
                return None
        else:
            raise ValueError(f"Unknown method: {method}")
