import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import sys

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pilot_net import PilotNet
from utils.data_loader import CarlaDataset
from utils.preprocessing import get_transforms

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loaders
    from utils.data_loader import get_dataloader
    
    use_mock = getattr(args, 'mock', False)
    if use_mock:
        print("WARNING: Using MOCK dataset for verification.")
    
    train_loader = get_dataloader(batch_size=args.batch_size, split="train", transform=get_transforms(train=True), use_mock=use_mock)
    val_loader = get_dataloader(batch_size=args.batch_size, split="test", transform=get_transforms(train=False), use_mock=use_mock)
    
    # Model
    model = PilotNet().to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, controls in loop:
            images = images.to(device)
            controls = controls.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, controls)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_outputs_list = []
        all_controls_list = []
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, controls in loop_val:
                images = images.to(device)
                controls = controls.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, controls)
                
                val_loss += loss.item()
                loop_val.set_postfix(loss=loss.item())
                
                all_outputs_list.append(outputs)
                all_controls_list.append(controls)
                
        avg_val_loss = val_loss / len(val_loader)
        
        if len(outputs) > 0 and len(controls) > 0:
            all_outputs = torch.cat([outputs for outputs in all_outputs_list], dim=0)
            all_controls = torch.cat([controls for controls in all_controls_list], dim=0)
            
            # Calculate detailed metrics
            from evaluation.metrics import calculate_metrics, calculate_f1_score
            calculate_metrics(all_outputs.cpu().numpy(), all_controls.cpu().numpy())
            calculate_f1_score(all_outputs.cpu().numpy(), all_controls.cpu().numpy())
            
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "pilot_net_best.pth"))
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_model(args)
