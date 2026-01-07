import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance

def get_transforms(width=200, height=66, train=True):
    """
    Returns torchvision transforms for preprocessing.
    Ref: NVidia PilotNet used 66x200 YUV images. 
    Here we stick to RGB for simplicity unless specified otherwise, but resize to 66x200.
    """
    transform_list = [
        transforms.Lambda(lambda img: img.convert('RGB')), # Ensure 3 channels
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        # Normalize roughly to [-1, 1] or [0, 1]. 
        # Standard ImageNet means are [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
        # But for driving, sometimes we just use 0.5 mean/std.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Simple augmentations for training
    if train:
        transform_list.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
    return transforms.Compose(transform_list)

def preprocess_image(image_path_or_pil):
    """
    Helper to load and preprocess a single image for inference.
    Accepts: file path (str), PIL Image, or numpy array.
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert('RGB')
    elif isinstance(image_path_or_pil, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_path_or_pil).convert('RGB')
    elif isinstance(image_path_or_pil, Image.Image):
        image = image_path_or_pil.convert('RGB')
    else:
        image = image_path_or_pil
        
    transform = get_transforms(train=False)
    return transform(image).unsqueeze(0) # Add batch dimension

def denormalize_image(tensor):
    """
    Convert normalized tensor back to PIL image for visualization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    to_pil = transforms.ToPILImage()
    return to_pil(tensor.squeeze(0) if len(tensor.shape) == 4 else tensor)
