import argparse
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.train import train_model
from inference.driver import Driver

def main():
    parser = argparse.ArgumentParser(description="Explainable Self-Driving Intelligence System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the PilotNet model")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4, dest="learning_rate")
    train_parser.add_argument("--save_dir", type=str, default="./models")
    train_parser.add_argument("--mock", action="store_true", help="Use mock dataset")
    
    # Drive Command
    drive_parser = subparsers.add_parser("drive", help="Run autonomous driving inference")
    drive_parser.add_argument("--model_path", type=str, default="models/pilot_net_best.pth")
    drive_parser.add_argument("--xai", type=str, default="grad_cam", choices=["grad_cam", "lime", "shap", "None"])
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("Starting training...")
        train_model(args)
    elif args.command == "drive":
        print(f"Starting autonomous driver with XAI: {args.xai}...")
        driver = Driver(args.model_path, xai_method=args.xai if args.xai != "None" else None)
        driver.run()

if __name__ == "__main__":
    main()
