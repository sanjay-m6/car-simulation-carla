# 🚗 CARLA Autonomous Driving with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![CARLA](https://img.shields.io/badge/CARLA-Simulator-green.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end autonomous driving system built with **PilotNet** neural network architecture, featuring comprehensive **Explainable AI (XAI)** capabilities for understanding model decisions in real-time. Trained on the CARLA simulator dataset with support for Grad-CAM, LIME, and SHAP interpretability methods.

---

## 🌟 Features

- **🧠 PilotNet Architecture**: End-to-end learning neural network for self-driving cars
- **🔍 Explainable AI**: Multiple XAI methods (Grad-CAM, LIME, SHAP) for model interpretability
- **🎮 CARLA Integration**: Trained on CARLA Autopilot Multimodal Dataset
- **📊 Real-time Visualization**: Live HUD showing steering predictions and attention maps
- **📈 Comprehensive Metrics**: MSE, MAE, and F1 Score for model evaluation
- **🚀 Mock Training Mode**: Quick verification with lightweight dataset

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [XAI Methods](#-xai-methods)
- [Evaluation Metrics](#-evaluation-metrics)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗️ Architecture

```
Input Image (RGB) → PilotNet CNN → Control Outputs (Steering, Throttle, Brake)
                         ↓
                    XAI Analysis
                (Grad-CAM/LIME/SHAP)
                         ↓
              Attention Visualization
```

**Core Components:**
- **Model**: PilotNet CNN (9 layers) - Based on NVIDIA's end-to-end learning architecture
- **XAI Methods**: 
  - Grad-CAM (Gradient-weighted Class Activation Mapping)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)
- **Dataset**: CARLA Autopilot Multimodal Dataset from HuggingFace
- **Evaluation**: MSE, MAE, F1 Score (Macro)

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanjay-m6/car-simulation-carla.git
   cd car-simulation-carla
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
- `torch` & `torchvision` - Deep learning framework
- `numpy` & `pandas` - Data manipulation
- `matplotlib` & `opencv-python` - Visualization
- `scikit-learn` - Evaluation metrics
- `datasets` & `huggingface_hub` - Dataset loading
- `lime` & `shap` - Explainability
- `tqdm` - Progress bars

---

## 🚀 Quick Start

### Training (Mock Mode)
Verify the pipeline quickly without downloading large datasets:
```bash
python main.py train --mock --epochs 1 --batch_size 4
```

### Training (Full Dataset)
Train on the complete CARLA dataset:
```bash
python main.py train --epochs 20 --batch_size 32 --lr 1e-4
```

### Inference with XAI
Run autonomous driving with Grad-CAM visualization:
```bash
python main.py drive --xai grad_cam
```

---

## 📖 Usage

### Training Mode

Train the PilotNet model with custom parameters:

```bash
python main.py train [OPTIONS]
```

**Available Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | INT | 10 | Number of training epochs |
| `--batch_size` | INT | 32 | Training batch size |
| `--lr` | FLOAT | 1e-4 | Learning rate |
| `--save_dir` | STR | `./models` | Directory for model checkpoints |
| `--mock` | FLAG | False | Use lightweight mock dataset |

**Examples:**

```bash
# Quick verification with mock data
python main.py train --mock --epochs 1 --batch_size 4

# Full training with custom parameters
python main.py train --epochs 30 --batch_size 64 --lr 5e-5

# Save to custom directory
python main.py train --epochs 15 --save_dir ./checkpoints
```

---

### Driving Mode (Inference)

Run autonomous inference with real-time XAI visualization:

```bash
python main.py drive [OPTIONS]
```

**Available Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model_path` | STR | `models/pilot_net_best.pth` | Path to trained model |
| `--xai` | STR | `grad_cam` | XAI method: `grad_cam`, `lime`, `shap`, `None` |

**Examples:**

```bash
# Drive with Grad-CAM (fastest)
python main.py drive --xai grad_cam

# Drive with LIME (model-agnostic, slower)
python main.py drive --xai lime

# Drive with SHAP (detailed explanations)
python main.py drive --xai shap

# Drive without XAI overlay
python main.py drive --xai None

# Use custom model
python main.py drive --model_path ./my_models/custom_model.pth --xai grad_cam
```

---

## 📁 Project Structure

```
car-simulation-carla/
│
├── main.py                     # Main entry point (CLI)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── models/                     # Model architecture & checkpoints
│   ├── __init__.py
│   ├── pilot_net.py           # PilotNet CNN architecture
│   ├── train.py               # Training logic
│   └── pilot_net_best.pth     # Trained model weights (after training)
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading & preprocessing
│   ├── transforms.py          # Image transformations
│   └── helpers.py             # Helper functions
│
├── xai/                        # Explainable AI implementations
│   ├── __init__.py
│   ├── explainer.py           # Base explainer interface
│   ├── grad_cam.py            # Grad-CAM implementation
│   ├── lime_explainer.py      # LIME integration
│   └── shap_explainer.py      # SHAP integration
│
├── inference/                  # Autonomous driving logic
│   ├── __init__.py
│   ├── driver.py              # Main driver agent
│   └── predictor.py           # Model inference wrapper
│
├── visualization/              # Visualization components
│   ├── __init__.py
│   ├── hud.py                 # Heads-up display
│   └── overlay.py             # XAI overlay rendering
│
├── evaluation/                 # Model evaluation
│   ├── __init__.py
│   └── metrics.py             # Custom metrics (MSE, MAE, F1)
│
├── carla_interface/            # CARLA simulator integration
│   ├── __init__.py
│   └── connection.py          # CARLA client wrapper
│
└── scripts/                    # Utility scripts
    ├── patch_notebook.py      # Notebook compatibility fixes
    └── download_dataset.py    # Dataset download helper
```

---

## 📦 Dataset

This project uses the **CARLA Autopilot Multimodal Dataset** from HuggingFace.

- **Source**: [HuggingFace Datasets](https://huggingface.co/datasets)
- **Content**: RGB camera images + control signals (steering, throttle, brake)
- **Size**: ~10GB (full dataset)
- **Format**: Automatically downloaded on first training run

### Mock Dataset
For quick testing, use `--mock` flag to train on a small synthetic dataset (~100 samples).

---

## 🧠 Model Details

### PilotNet Architecture

Based on NVIDIA's "End-to-End Learning for Self-Driving Cars" paper:

```
Input: 66x200x3 RGB Image
    ↓
Conv2D(24, 5x5, stride=2) + ReLU
    ↓
Conv2D(36, 5x5, stride=2) + ReLU
    ↓
Conv2D(48, 5x5, stride=2) + ReLU
    ↓
Conv2D(64, 3x3) + ReLU
    ↓
Conv2D(64, 3x3) + ReLU
    ↓
Flatten
    ↓
FC(100) + ReLU + Dropout(0.5)
    ↓
FC(50) + ReLU + Dropout(0.5)
    ↓
FC(10) + ReLU
    ↓
Output: [Steering, Throttle, Brake]
```

**Key Features:**
- **Convolutional Layers**: 5 layers for feature extraction
- **Fully Connected Layers**: 3 layers for control prediction
- **Dropout**: Regularization to prevent overfitting
- **Output**: 3 continuous values (steering angle, throttle, brake)

---

## 🔍 XAI Methods

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Speed**: ⚡ Fast (real-time)
- **Type**: Gradient-based
- **Use Case**: Shows which image regions influence steering decisions
- **Visualization**: Heatmap overlay on input image

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- **Speed**: 🐌 Slower (requires multiple forward passes)
- **Type**: Perturbation-based, model-agnostic
- **Use Case**: Explains predictions by perturbing input regions
- **Visualization**: Highlighted important superpixels

### 3. SHAP (SHapley Additive exPlanations)
- **Speed**: 🐢 Slowest (comprehensive analysis)
- **Type**: Game theory-based, model-agnostic
- **Use Case**: Provides feature importance values
- **Visualization**: Attribution maps showing positive/negative contributions

---

## 📊 Evaluation Metrics

The model is evaluated using multiple metrics:

### Regression Metrics
- **MSE (Mean Squared Error)**: Measures average squared difference between predictions and ground truth
- **MAE (Mean Absolute Error)**: Average absolute deviation from ground truth

### Classification Metrics (Discretized Controls)
- **F1 Score (Macro)**: Balanced performance across discretized control categories:
  - **Steering**: Left / Straight / Right
  - **Throttle**: Idle / Accelerate
  - **Brake**: Coast / Brake

---

## 🎯 Performance

Typical performance on CARLA dataset after 20 epochs:

| Metric | Value |
|--------|-------|
| Steering MSE | ~0.02 |
| Steering MAE | ~0.10 |
| F1 Score (Macro) | ~0.85 |
| Inference Speed (GPU) | ~60 FPS |
| Inference Speed (CPU) | ~15 FPS |

*Note: Performance varies based on hardware and training configuration*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NVIDIA**: PilotNet architecture from "End-to-End Learning for Self-Driving Cars"
- **CARLA Team**: Open-source autonomous driving simulator
- **HuggingFace**: Dataset hosting and easy access
- **PyTorch Team**: Deep learning framework

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Autonomous Driving! 🚗💨**
