import os
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, dataset_name="immanuelpeter/carla-autopilot-multimodal-dataset", split="train", transform=None, cache_dir="./data_cache"):
        """
        Args:
            dataset_name (str): HuggingFace dataset name.
            split (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            cache_dir (str): Directory to store downloaded data.
        """
        self.transform = transform
        self.cache_dir = cache_dir
        
        print(f"Loading dataset {dataset_name} ({split})...")
        # Load dataset from HuggingFace
        # streaming=True allows us to load without downloading everything at once, 
        # but for training stability, downloading might be better if size permits.
        # Assuming the dataset fits on disk, we load it.
        try:
            self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e
            
        print(f"Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Dataset columns (from HuggingFace):
        # - image_front: RGB front camera image
        # - steer: steering value
        # - throttle: throttle value  
        # - brake: brake value
        
        image = item.get('image_front')
        if image is None:
            # Fallback to 'image' if 'image_front' doesn't exist
            image = item.get('image')
            
        if not isinstance(image, Image.Image):
            # If it's bytes or something else, handle it. 
            # HF datasets usually return PIL Images for image feature types.
            pass
            
        # Use 'steer' not 'steering' per the dataset schema
        steering = float(item.get('steer', 0.0))
        throttle = float(item.get('throttle', 0.0))
        brake = float(item.get('brake', 0.0))
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and the control vector
        controls = torch.tensor([steering, throttle, brake], dtype=torch.float32)
        
        return image, controls

        return image, controls

class MockCarlaDataset(Dataset):
    "Mock dataset for testing pipeline without downloading data."
    def __init__(self, length=100, transform=None):
        self.length = length
        self.transform = transform
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Generate random image (66, 200, 3)
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

def get_dataloader(batch_size=32, split="train", transform=None, use_mock=False):
    if use_mock:
        dataset = MockCarlaDataset(transform=transform)
    else:
        dataset = CarlaDataset(split=split, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train" or use_mock))

if __name__ == "__main__":
    # Test the loader
    ds = CarlaDataset(split="train")
    img, ctrl = ds[0]
    print(f"Sample 0 controls: {ctrl}")
    print(f"Image size: {img.size}")
