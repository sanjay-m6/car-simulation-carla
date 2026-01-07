import cv2
from lime import lime_image
import torch
import numpy as np
from skimage.segmentation import mark_boundaries

class LimeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def _predict_fn(self, images_np):
        """
        Wrapper for model prediction.
        images_np: (N, H, W, 3) or (N, H, W) numpy array, range [0, 1] or [0, 255]?
        Usually LIME passes double range.
        We need to preprocess to tensor (N, 3, 66, 200).
        """
        # Assume images_np is (N, 66, 200, 3) range depends on usage.
        # But our model expects (N, 3, 66, 200).
        
        self.model.eval()
        
        # Preprocess
        # Step 1: To tensor
        tensors = []
        for img in images_np:
            # img is numpy array. Convert to tensor.
            # Assuming img is [0, 1] double or [0, 255] uint8.
            if img.max() > 1:
                img = img / 255.0
            
            # Resize to model input size (200, 66)
            # The model expects (3, 66, 200)
            # cv2.resize expects (W, H)
            img_resized = cv2.resize(img, (200, 66))
            
            # (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
            
            # Normalize (as per our preprocessing)
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            
            tensors.append(tensor)
            
        batch = torch.stack(tensors)
        if torch.cuda.is_available():
            batch = batch.cuda()
            
        with torch.no_grad():
            outputs = self.model(batch) # (N, 3)
            
        return outputs.cpu().numpy()

    def explain(self, image_np, target_index=0, num_samples=1000):
        """
        image_np: (H, W, 3)
        target_index: 0, 1, 2
        """
        # Ensure 3 channels (RGB) if 4 (RGBA)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        explanation = self.explainer.explain_instance(
            image_np.astype('double'), 
            self._predict_fn, 
            top_labels=None, 
            num_features=1000, 
            num_samples=num_samples,
            labels=(target_index,) # For regression, providing labels is tricky.
            # If mode is regression, labels argument might be ignored or handled differently?
            # Actually lime_image by default does classification. 
            # We might need to specify mode='regression' in LimeImageExplainer?
            # It seems LimeImageExplainer doesn't have mode arg in init, but explain_instance determines?
            # Actually we pass specific labels we want to explain.
        )
        
        # visualize
        temp, mask = explanation.get_image_and_mask(
            target_index, 
            positive_only=False, 
            num_features=10, 
            hide_rest=False
        )
        # temp is the image with superpixels?
        # mask is the superpixels mask?
        
        # We can also get the mask directly.
        # But let's return the overlay.
        return mark_boundaries(temp / 2 + 0.5, mask)
