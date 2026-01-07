# 🔄 CARLA Autonomous Driving - Project Workflow

This document provides a comprehensive overview of the project's workflow, including system architecture, training pipeline, and inference execution.

---

## 📊 Table of Contents

- [System Overview](#-system-overview)
- [Training Workflow](#-training-workflow)
- [Inference Workflow](#-inference-workflow)
- [XAI Pipeline](#-xai-pipeline)
- [Component Interaction](#-component-interaction)
- [Data Flow](#-data-flow)

---

## 🎯 System Overview

```mermaid
graph TB
    A[User] -->|Train Command| B[Training Pipeline]
    A -->|Drive Command| C[Inference Pipeline]
    
    B --> D[Data Loader]
    D --> E[CARLA Dataset]
    D --> F[Mock Dataset]
    
    B --> G[PilotNet Model]
    G --> H[Model Checkpoint]
    
    C --> H
    C --> I[XAI Engine]
    I --> J[Grad-CAM]
    I --> K[LIME]
    I --> L[SHAP]
    
    C --> M[Visualization HUD]
    M --> N[Real-time Display]
```

**Key Components:**
- **Training Pipeline**: Handles model training and evaluation
- **Inference Pipeline**: Runs autonomous driving with predictions
- **XAI Engine**: Generates explanations for model decisions
- **Visualization**: Real-time HUD with attention maps

---

## 🎓 Training Workflow

### High-Level Training Flow

```mermaid
flowchart TD
    Start([python main.py train]) --> Parse[Parse Arguments]
    Parse --> Mock{Mock Mode?}
    
    Mock -->|Yes| MockData[Generate Mock Dataset]
    Mock -->|No| Download[Download CARLA Dataset]
    
    MockData --> Preprocess[Data Preprocessing]
    Download --> Preprocess
    
    Preprocess --> Split[Train/Val Split]
    Split --> Transform[Apply Transforms]
    Transform --> Loader[Create DataLoaders]
    
    Loader --> InitModel[Initialize PilotNet]
    InitModel --> InitOpt[Setup Optimizer & Loss]
    
    InitOpt --> TrainLoop{Training Epochs}
    
    TrainLoop --> Forward[Forward Pass]
    Forward --> CalcLoss[Calculate Loss]
    CalcLoss --> Backward[Backward Pass]
    Backward --> UpdateWeights[Update Weights]
    
    UpdateWeights --> Validate[Validation]
    Validate --> Metrics[Calculate Metrics]
    Metrics --> Log[Log Progress]
    
    Log --> CheckBest{Best Model?}
    CheckBest -->|Yes| SaveBest[Save Best Model]
    CheckBest -->|No| Continue
    SaveBest --> Continue[Continue Training]
    
    Continue --> TrainLoop
    
    TrainLoop -->|Epochs Complete| Final[Save Final Checkpoint]
    Final --> End([Training Complete])
    
    style Start fill:#4CAF50
    style End fill:#2196F3
    style SaveBest fill:#FF9800
```

### Detailed Training Steps

1. **Initialization**
   - Parse command-line arguments
   - Set device (GPU/CPU)
   - Initialize random seeds for reproducibility

2. **Data Loading**
   ```
   Option A: Mock Mode
   └── Generate synthetic dataset (100 samples)
   
   Option B: Full Dataset
   ├── Download CARLA dataset from HuggingFace
   ├── Cache locally for future runs
   └── Load training and validation splits
   ```

3. **Preprocessing**
   - Resize images to 66x200 (PilotNet input size)
   - Normalize pixel values to [0, 1]
   - Convert to PyTorch tensors
   - Normalize controls (steering, throttle, brake)

4. **Model Training Loop**
   ```
   For each epoch:
   ├── Training Phase
   │   ├── Forward pass: predictions = model(images)
   │   ├── Compute loss: MSE(predictions, ground_truth)
   │   ├── Backward pass: loss.backward()
   │   └── Update weights: optimizer.step()
   │
   └── Validation Phase
       ├── Evaluate on validation set
       ├── Calculate metrics (MSE, MAE, F1)
       └── Save best model if improved
   ```

5. **Metrics Calculation**
   - **MSE**: Mean Squared Error for regression accuracy
   - **MAE**: Mean Absolute Error for average deviation
   - **F1 Score**: Discretized classification performance

6. **Model Saving**
   - Best model: Saved when validation loss improves
   - Final model: Saved at end of training
   - Checkpoints include: model weights, optimizer state, epoch number

---

## 🚗 Inference Workflow

### High-Level Inference Flow

```mermaid
flowchart TD
    Start([python main.py drive]) --> ParseArgs[Parse Arguments]
    ParseArgs --> LoadModel[Load Trained Model]
    
    LoadModel --> InitXAI{XAI Method?}
    
    InitXAI -->|grad_cam| GradCAM[Initialize Grad-CAM]
    InitXAI -->|lime| LIME[Initialize LIME]
    InitXAI -->|shap| SHAP[Initialize SHAP]
    InitXAI -->|None| NoXAI[No XAI]
    
    GradCAM --> InitDriver[Initialize Driver Agent]
    LIME --> InitDriver
    SHAP --> InitDriver
    NoXAI --> InitDriver
    
    InitDriver --> LoadData[Load Test Dataset]
    LoadData --> DriveLoop{For Each Frame}
    
    DriveLoop --> Preprocess[Preprocess Image]
    Preprocess --> Predict[Model Prediction]
    
    Predict --> XAIGen{Generate XAI?}
    
    XAIGen -->|Yes| GenExplanation[Generate Explanation]
    XAIGen -->|No| SkipXAI[Skip XAI]
    
    GenExplanation --> CreateHeatmap[Create Attention Heatmap]
    CreateHeatmap --> Overlay
    SkipXAI --> Overlay[Overlay on Image]
    
    Overlay --> DisplayHUD[Display HUD]
    DisplayHUD --> ShowControls[Show Predicted Controls]
    
    ShowControls --> DriveLoop
    
    DriveLoop -->|Complete| End([Inference Complete])
    
    style Start fill:#4CAF50
    style End fill:#2196F3
    style Predict fill:#FF9800
    style GenExplanation fill:#9C27B0
```

### Detailed Inference Steps

1. **Model Loading**
   ```python
   model = PilotNet()
   checkpoint = torch.load(model_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. **XAI Initialization**
   - **Grad-CAM**: Extract target convolutional layer
   - **LIME**: Initialize image explainer with segmentation
   - **SHAP**: Create deep explainer with background samples

3. **Frame Processing Loop**
   ```
   For each frame:
   ├── Load image from dataset
   ├── Preprocess (resize, normalize)
   ├── Model inference: controls = model(image)
   ├── Generate XAI explanation (if enabled)
   ├── Create visualization overlay
   └── Display on HUD
   ```

4. **Prediction Extraction**
   ```python
   outputs = model(image_tensor)
   steering = outputs[0, 0]
   throttle = outputs[0, 1]
   brake = outputs[0, 2]
   ```

5. **Real-time Display**
   - Original image with XAI overlay
   - Steering wheel indicator
   - Numerical control values
   - FPS counter

---

## 🔍 XAI Pipeline

### XAI Methods Comparison

```mermaid
graph LR
    A[Input Image] --> B{XAI Method}
    
    B -->|Grad-CAM| C[Extract Gradients]
    C --> D[Compute Activation Map]
    D --> E[Generate Heatmap]
    
    B -->|LIME| F[Segment Image]
    F --> G[Perturb Segments]
    G --> H[Fit Linear Model]
    H --> I[Highlight Important Regions]
    
    B -->|SHAP| J[Sample Backgrounds]
    J --> K[Compute Shapley Values]
    K --> L[Attribution Map]
    
    E --> M[Overlay on Image]
    I --> M
    L --> M
    
    M --> N[Visualization Output]
    
    style C fill:#FF5722
    style F fill:#2196F3
    style J fill:#4CAF50
```

### Grad-CAM Workflow

```mermaid
sequenceDiagram
    participant I as Input Image
    participant M as PilotNet Model
    participant L as Target Layer
    participant G as Grad-CAM
    participant V as Visualization
    
    I->>M: Forward pass
    M->>L: Extract activations
    M->>M: Compute steering prediction
    M->>L: Backward to target layer
    L->>G: Get gradients
    G->>G: Global average pooling
    G->>G: Weight activations
    G->>G: ReLU (positive influence)
    G->>V: Generate heatmap
    V->>V: Resize to input size
    V->>V: Overlay on original image
```

### LIME Workflow

```mermaid
sequenceDiagram
    participant I as Input Image
    participant S as Segmentation
    participant P as Perturbation
    participant M as Model
    participant E as Linear Explainer
    participant V as Visualization
    
    I->>S: Segment into superpixels
    S->>P: Create perturbed samples
    P->>M: Predict on samples
    M->>E: Return predictions
    E->>E: Fit linear model
    E->>E: Extract feature importance
    E->>V: Highlight important segments
    V->>V: Create explanation mask
```

### SHAP Workflow

```mermaid
sequenceDiagram
    participant D as Dataset
    participant B as Background Samples
    participant I as Input Image
    participant S as SHAP
    participant M as Model
    participant V as Visualization
    
    D->>B: Sample background data
    B->>S: Initialize explainer
    I->>S: Input test image
    S->>M: Multiple forward passes
    M->>S: Return predictions
    S->>S: Compute Shapley values
    S->>S: Calculate attribution
    S->>V: Generate attribution map
    V->>V: Visualize pixel contributions
```

---

## 🔗 Component Interaction

```mermaid
graph TB
    subgraph "Entry Point"
        Main[main.py]
    end
    
    subgraph "Core Models"
        PN[PilotNet<br/>models/pilot_net.py]
        Train[Trainer<br/>models/train.py]
    end
    
    subgraph "Data Pipeline"
        DL[DataLoader<br/>utils/data_loader.py]
        Trans[Transforms<br/>utils/transforms.py]
    end
    
    subgraph "XAI Components"
        GC[Grad-CAM<br/>xai/grad_cam.py]
        LI[LIME<br/>xai/lime_explainer.py]
        SH[SHAP<br/>xai/shap_explainer.py]
        Exp[Base Explainer<br/>xai/explainer.py]
    end
    
    subgraph "Inference Engine"
        Driver[Driver Agent<br/>inference/driver.py]
        Pred[Predictor<br/>inference/predictor.py]
    end
    
    subgraph "Visualization"
        HUD[HUD Display<br/>visualization/hud.py]
        Overlay[XAI Overlay<br/>visualization/overlay.py]
    end
    
    subgraph "Evaluation"
        Met[Metrics<br/>evaluation/metrics.py]
    end
    
    Main -->|train| Train
    Main -->|drive| Driver
    
    Train --> PN
    Train --> DL
    Train --> Met
    
    DL --> Trans
    
    Driver --> Pred
    Driver --> HUD
    
    Pred --> PN
    Pred --> Exp
    
    Exp --> GC
    Exp --> LI
    Exp --> SH
    
    HUD --> Overlay
    Overlay --> GC
    Overlay --> LI
    Overlay --> SH
    
    style Main fill:#4CAF50
    style PN fill:#FF9800
    style Driver fill:#2196F3
```

---

## 📈 Data Flow

### Training Data Flow

```mermaid
flowchart LR
    A[(CARLA Dataset<br/>HuggingFace)] --> B[Download & Cache]
    B --> C[data_cache/]
    
    C --> D[DataLoader]
    D --> E[Image Transform]
    D --> F[Label Normalization]
    
    E --> G[Batch: 66x200x3]
    F --> H[Controls: steering, throttle, brake]
    
    G --> I[PilotNet CNN]
    H --> I
    
    I --> J[Loss Calculation]
    J --> K[Backpropagation]
    K --> L[Weight Update]
    
    L --> M[Validation]
    M --> N{Improved?}
    
    N -->|Yes| O[Save Checkpoint<br/>models/pilot_net_best.pth]
    N -->|No| P[Continue Training]
    
    style A fill:#4CAF50
    style I fill:#FF9800
    style O fill:#2196F3
```

### Inference Data Flow

```mermaid
flowchart LR
    A[Test Dataset] --> B[Load Frame]
    B --> C[Preprocess<br/>66x200x3]
    
    C --> D[PilotNet<br/>Forward Pass]
    
    D --> E[Steering Prediction]
    D --> F[Throttle Prediction]
    D --> G[Brake Prediction]
    
    C --> H{XAI Enabled?}
    
    H -->|Grad-CAM| I[Extract Gradients]
    H -->|LIME| J[Perturb Image]
    H -->|SHAP| K[Shapley Analysis]
    H -->|None| L[Skip XAI]
    
    I --> M[Heatmap]
    J --> M
    K --> M
    
    M --> N[Overlay on Image]
    L --> N
    
    E --> O[HUD Display]
    F --> O
    G --> O
    N --> O
    
    O --> P[Screen Output]
    
    style D fill:#FF9800
    style M fill:#9C27B0
    style P fill:#4CAF50
```

---

## 🎮 Usage Flow

### Quick Start Flow

```mermaid
stateDiagram-v2
    [*] --> Clone: git clone
    Clone --> Install: pip install -r requirements.txt
    
    Install --> TrainChoice: Choose Mode
    
    state TrainChoice <<choice>>
    TrainChoice --> MockTrain: Quick Test
    TrainChoice --> FullTrain: Full Training
    
    MockTrain --> ModelReady: 1-2 minutes
    FullTrain --> ModelReady: 30-60 minutes
    
    ModelReady --> InferenceChoice: Choose XAI
    
    state InferenceChoice <<choice>>
    InferenceChoice --> GradCAMDrive: Grad-CAM (Fast)
    InferenceChoice --> LIMEDrive: LIME (Slow)
    InferenceChoice --> SHAPDrive: SHAP (Slower)
    InferenceChoice --> NoDrive: No XAI
    
    GradCAMDrive --> [*]
    LIMEDrive --> [*]
    SHAPDrive --> [*]
    NoDrive --> [*]
```

---

## 📋 Command Flow Summary

### Training Command Flow
```bash
python main.py train --epochs 10 --batch_size 32
    ↓
Parse arguments
    ↓
Load/Download dataset
    ↓
Initialize PilotNet model
    ↓
Training loop (10 epochs)
    ↓
Save best model → models/pilot_net_best.pth
```

### Inference Command Flow
```bash
python main.py drive --xai grad_cam
    ↓
Load trained model
    ↓
Initialize Grad-CAM explainer
    ↓
Load test images
    ↓
For each frame:
    - Predict controls
    - Generate heatmap
    - Display HUD
```

---

## 🔄 Continuous Workflow

```mermaid
graph TB
    A[Develop] --> B[Train Model]
    B --> C[Evaluate Performance]
    C --> D{Satisfactory?}
    
    D -->|No| E[Adjust Hyperparameters]
    E --> B
    
    D -->|Yes| F[Test Inference]
    F --> G[Verify XAI]
    
    G --> H{XAI Clear?}
    H -->|No| I[Debug Model]
    I --> B
    
    H -->|Yes| J[Deploy/Share]
    J --> K[Collect Feedback]
    K --> L{Improvements Needed?}
    
    L -->|Yes| A
    L -->|No| M[Maintain]
    
    style B fill:#4CAF50
    style F fill:#2196F3
    style J fill:#FF9800
```

---

## 🎯 Key Takeaways

1. **Modular Design**: Each component (model, XAI, visualization) is independent
2. **Two Main Modes**: Training (batch processing) and Inference (real-time)
3. **Flexible XAI**: Choose from multiple explanation methods based on needs
4. **Efficient Pipeline**: Mock mode for quick testing, full mode for production
5. **Clear Data Flow**: Raw data → Processing → Model → Predictions → Visualization

---

**For more details on specific components, refer to the [README.md](README.md)**
