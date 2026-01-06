# Breast Cancer Detection Using CNN

## Overview
This project implements a deep learning-based breast cancer classification system using histopathological images. The system employs transfer learning with pre-trained CNN architectures to distinguish between benign and malignant breast tissue samples. Two different approaches are demonstrated: one following a research paper implementation (MobileNetV3), and another enhanced version using EfficientNet-B0.

## Features
- **Binary Classification**: Benign vs. Malignant tissue detection
- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Data Augmentation**: Extensive augmentation for improved generalization
- **Early Stopping**: Prevents overfitting with patience-based training termination
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Comprehensive Evaluation**: Training/validation/test split with detailed metrics

## Technologies Used
- **PyTorch**: Deep learning framework
- **TorchVision**: Pre-trained models and image transformations
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization of training metrics
- **Scikit-learn**: Data splitting and evaluation
- **Google Colab**: GPU-accelerated training environment

## Dataset
- **Source**: Breast Cancer Histopathological Database
- **Classes**: 
  - Benign (2,480 images)
  - Malignant (5,429 images)
- **Total**: 7,909 images
- **Split**: 
  - Training: 75% (5,931 images)
  - Validation: 15% (1,186 images)
  - Test: 10% (792 images)
- **Image Format**: Histopathological slides at 224×224 resolution

## Model Architectures

### 1. Paper Implementation (MobileNetV3 Large)

#### Architecture Details
- **Base Model**: MobileNetV3 Large (ImageNet pre-trained)
- **Input Size**: 224×224×3
- **Transfer Learning Strategy**: 
  - First 150 parameter groups frozen
  - Remaining layers fine-tuned

#### Custom Classifier
```python
nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(960, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)
```

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Learning Rate Decay | Exponential (γ=0.95) |
| Batch Size | 32 |
| Epochs | 13 (fixed) |
| Loss Function | CrossEntropyLoss |

#### Data Augmentation (Training)
- Resize to 224×224
- Random horizontal flip
- Normalization (ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2. Enhanced Implementation (EfficientNet-B0)

#### Architecture Details
- **Base Model**: EfficientNet-B0 (ImageNet pre-trained)
- **Input Size**: 224×224×3
- **Transfer Learning Strategy**:
  - All early layers frozen
  - Last 2 feature blocks unfrozen for fine-tuning

#### Custom Classifier
```python
nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.4),
    nn.Linear(512, 2)
)
```

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (with weight decay) |
| Initial Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Learning Rate Scheduler | ReduceLROnPlateau (factor=0.1, patience=3) |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping Patience | 7 |
| Loss Function | CrossEntropyLoss |

#### Enhanced Data Augmentation (Training)
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.1, contrast=0.1)
- Normalization (ImageNet statistics)


## Training Process

### Paper Implementation (MobileNetV3)
- **Training Time**: 19.50 minutes
- **Fixed Epochs**: 13
- **Learning Rate Schedule**: Exponential decay (0.95 per epoch)
- **Epoch-by-Epoch Results**:
  - Epoch 1: Train Acc 85.35%, Val Acc 88.11%
  - Epoch 8: Train Acc 98.65%, Val Acc 97.13%
  - Epoch 12: Train Acc 99.11%, Val Acc 97.72%
  - Epoch 13: Train Acc 99.06%, Val Acc 96.63%
- **Final Test Accuracy**: 95.45%

### Enhanced Implementation (EfficientNet-B0)
- **Training**: Stopped at epoch 26 (early stopping triggered)
- **Best Validation Performance**: Epoch 19
  - Validation Loss: 0.0606
  - Validation Accuracy: 97.47%
- **Training Progression**:
  - Epoch 1: Train Acc 84.32%, Val Acc 91.99%
  - Epoch 10: Train Acc 96.29%, Val Acc 95.95%
  - Epoch 15: Train Acc 97.03%, Val Acc 97.13%
  - Epoch 19: Train Acc 97.00%, Val Acc 97.47% (Best)
  - Epoch 26: Early stopping triggered
- **Final Test Accuracy**: 98.11%

## Results Comparison

| Metric | MobileNetV3 (Paper) | EfficientNet-B0 (Enhanced) |
|--------|---------------------|---------------------------|
| Training Accuracy | 99.06% | 98.23% (epoch 25) |
| Validation Accuracy | 96.63% | 97.47% (best) |
| Test Accuracy | 95.45% | **98.11%** |
| Training Time | 19.5 min | ~Variable (early stop) |
| Epochs Completed | 13 | 26 |
| Overfitting Gap | 2.43% | 0.76% |

### Key Observations
1. **Enhanced Model Performance**: EfficientNet-B0 achieved 2.66% higher test accuracy
2. **Better Generalization**: Smaller gap between training and test accuracy (98.23% → 98.11% vs 99.06% → 95.45%)
3. **Improved Regularization**: 
   - Higher dropout rates (0.3, 0.4 vs 0.2)
   - Weight decay (1e-4) in AdamW optimizer
   - More aggressive data augmentation (vertical flip, rotation, color jitter)
4. **Adaptive Training**: Early stopping prevented unnecessary epochs and overfitting
5. **Robust Learning**: ReduceLROnPlateau scheduler allowed finer optimization near convergence



## Fine-Tuning Strategy

### Layer Freezing Rationale
- **MobileNetV3**: First 150 parameter groups frozen
  - Preserves low-level features learned from ImageNet (edges, textures, basic patterns)
  - Adapts high-level features to medical imaging domain
  - Reduces computational cost and training time
  
- **EfficientNet-B0**: Last 2 feature blocks unfrozen
  - More selective fine-tuning approach
  - Balances pre-trained knowledge retention and task-specific learning
  - Compound scaling architecture benefits from targeted unfreezing

### Optimization Techniques

#### 1. Optimizer Selection
- **Adam (MobileNetV3)**: Standard adaptive learning rate optimizer
- **AdamW (EfficientNet)**: Decoupled weight decay regularization
  - Better generalization through explicit L2 regularization
  - Weight decay = 1e-4

#### 2. Learning Rate Scheduling
- **MobileNetV3**: Exponential decay
```python
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
  # LR reduces: 0.001 → 0.00095 → 0.00090 ... (per epoch)
```
  
- **EfficientNet**: ReduceLROnPlateau
```python
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.1, patience=3
  )
  # Reduces LR by 10× when validation loss plateaus for 3 epochs
```

#### 3. Regularization Techniques
- **Dropout**: Randomly drops neurons during training
  - MobileNetV3: 0.2 (20%)
  - EfficientNet: 0.3 and 0.4 (30% and 40%) for stronger regularization
  
- **Batch Normalization**: 
  - Normalizes layer inputs
  - Stabilizes training, allows higher learning rates
  - Reduces internal covariate shift
  
- **Data Augmentation**: 
  - Artificially expands dataset
  - Forces model to learn invariant features
  - Enhanced version includes 5 augmentation types vs. 1 in paper

#### 4. Early Stopping
```python
patience = 7  # Wait 7 epochs for improvement
if val_loss improves:
    save_best_model()
    counter = 0
else:
    counter += 1
    if counter >= patience:
        stop_training()
```

