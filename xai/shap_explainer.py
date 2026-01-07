import shap
import torch
import numpy as np

class ShapExplainer:
    def __init__(self, model, background_data):
        """
        model: PyTorch model
        background_data: Tensor (N, 3, H, W) to use as background for DeepExplainer/GradientExplainer
        """
        self.model = model
        self.background_data = background_data
        # GradientExplainer is suitable for PyTorch models and image data
        self.explainer = shap.GradientExplainer(model, background_data)

    def explain(self, input_tensor, target_index=0):
        """
        input_tensor: (1, 3, H, W)
        target_index: Which output to explain.
        """
        # SHAP values will be a list of tensors (one for each output)
        # or a tensor if single output.
        shap_values = self.explainer.shap_values(input_tensor)
        
        # shap_values is list of length 3 (st, th, br)
        # each element is (1, 3, H, W)
        
        if isinstance(shap_values, list):
            vals = shap_values[target_index]
        else:
            vals = shap_values
            
        # Sum over color channels to get 2D heatmap? 
        # Or usually we visualize (H, W, 3).
        # We can return the raw shap values and let the visualizer handle it 
        # using shap.image_plot but that requires matplotlib.
        # For overlay, we usually take sum of absolute values across channels.
        
        vals = np.abs(vals).sum(axis=1) # (1, H, W)
        vals = vals.squeeze() # (H, W)
        
        # Normalize
        if vals.max() > 0:
            vals = vals / vals.max()
            
        return vals
