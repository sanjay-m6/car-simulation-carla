from captum.attr import LayerGradCam, LayerAttribution
import torch
import numpy as np
import cv2

class GradCamExplainer:
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model.
            target_layer: The layer to target for Grad-CAM (e.g., model.conv5)
        """
        self.model = model
        self.target_layer = target_layer
        self.grad_cam = LayerGradCam(model, target_layer)

    def explain(self, input_tensor, target_index=0):
        """
        Args:
            input_tensor: (1, C, H, W)
            target_index: Output index to explain (0=Steering, 1=Throttle, 2=Brake)
        Returns:
            heatmap: Normalized heatmap (H, W)
        """
        # Compute attribution
        # attribute returns tensor of same shape as layer output.
        # But LayerGradCam returns upsampled attribution to input size usually if attribute_to_layer_input=False?
        # Captum LayerGradCam returns attributions of the shape of the layer output, 
        # but we usually want it upsampled to input size.
        # Actually LayerGradCam returns spatial attribution map at the layer's resolution.
        
        attributions = self.grad_cam.attribute(input_tensor, target=target_index)
        
        # Upsample to input size
        # attributions shape: (1, 1, H_layer, W_layer)
        desired_h, desired_w = input_tensor.shape[2], input_tensor.shape[3]
        
        attr_upsampled = LayerAttribution.interpolate(attributions, (desired_h, desired_w), interpolate_mode='bilinear')
        
        # Squeeze and normalize
        heatmap = attr_upsampled.detach().cpu().squeeze().numpy()
        
        # ReLU (Grad-CAM keeps only positive influence, although Captum does this internally often, let's ensure)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap

    def visualize(self, original_image_np, heatmap):
        """
        Overlays heatmap on original image.
        original_image_np: (H, W, 3) range [0, 255] or [0,1]
        heatmap: (H, W) range [0, 1]
        """
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if np.max(original_image_np) <= 1.0:
            original_image_np = (original_image_np * 255).astype(np.uint8)
            
        # Ensure 3 channels (RGB) if 4 (RGBA)
        if len(original_image_np.shape) == 3 and original_image_np.shape[2] == 4:
             original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGBA2RGB)
            
        overlay = cv2.addWeighted(original_image_np, 0.6, heatmap_colored, 0.4, 0)
        return overlay
