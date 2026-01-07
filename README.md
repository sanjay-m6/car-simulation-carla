# Explainable Self-Driving Intelligence System

This project implements a self-driving car model (PilotNet) with Explainable AI (XAI) capabilities. It supports training on the CARLA dataset and running autonomous inference with real-time visualization of model attention.

## Architecture
- **Model**: PilotNet (End-to-End Learning for Self-Driving Cars)
- **XAI**: Grad-CAM, LIME, SHAP
- **Dataset**: CARLA Autopilot Multimodal Dataset
- **Evaluation**: MSE, MAE, F1 Score (Macro)

## Installation

1.  **Clone the repository** (if not already done).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `opencv-python`, `scikit-learn`, `tqdm`, `datasets`, `huggingface_hub`, `lime`, `shap` installed.*

## Usage

 The main entry point is `main.py`. It supports two modes: `train` and `drive`.

### 1. Training

Train the PilotNet model on the dataset.

**Command:**
```bash
python main.py train [OPTIONS]
```

**Options:**
- `--epochs INT`: Number of training epochs (default: 10).
- `--batch_size INT`: Batch size (default: 32).
- `--lr FLOAT`: Learning rate (default: 1e-4).
- `--save_dir STR`: Directory to save model checkpoints (default: `./models`).
- `--mock`: **[NEW]** Use a lightweight mock dataset for verification (skips large downloads).

**Examples:**
```bash
# Verify the pipeline quickly with mock data
python main.py train --mock --epochs 1 --batch_size 4

# Run full training (downloading dataset might take time)
python main.py train --epochs 20 --batch_size 64
```

### 2. Autonomous Driving (Inference)

Run the autonomous driver agent with XAI visualization.

**Command:**
```bash
python main.py drive [OPTIONS]
```

**Options:**
- `--model_path STR`: Path to the trained model (default: `models/pilot_net_best.pth`).
- `--xai STR`: Explainability method to visualize. Choices: `grad_cam`, `lime`, `shap`, `None` (default: `grad_cam`).

**Examples:**
```bash
# Drive using Grad-CAM explanations
python main.py drive --xai grad_cam

# Drive using LIME (slower but model-agnostic)
python main.py drive --xai lime

# Drive without XAI
python main.py drive --xai None
```

## Metrics
The training process reports:
- **MSE (Mean Squared Error)**: For regression accuracy.
- **MAE (Mean Absolute Error)**: For average deviation.
- **F1 Score (Macro)**: Discretized classification performance for Steering (Left/Straight/Right), Throttle (Idle/Accel), and Brake (Coast/Brake).
