# mini-torch: A PyTorch-like Deep Learning Library

A minimal deep learning library implemented from scratch using only NumPy and Python. This project reimplements the essential features of PyTorch including automatic differentiation, neural network layers, optimizers, and more.

## Project Overview

This project was created as an educational implementation to understand the internals of modern deep learning frameworks. It includes:

- **Automatic Differentiation**: Full reverse-mode autodiff with computational graph
- **Batched Tensor Operations**: Support for multidimensional arrays with broadcasting
- **Neural Network Layers**: Conv2D, Linear, BatchNorm, LayerNorm, Pooling, Dropout
- **Optimizers**: SGD, Momentum, Adagrad, RMSProp, Adam, AdamW, Adadelta (all complete)
- **Learning Rate Schedulers**: StepLR, ExponentialLR, CosineAnnealing, ReduceLROnPlateau
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, and more
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy
- **Advanced Architectures**: ResNet, Multi-Head Attention, Transformers

## Project Structure

```
mini-torch/
├── mini_torch/              # Core library
│   ├── __init__.py         # Package initialization
│   ├── tensor.py           # Batched Tensor with autograd
│   ├── nn.py               # Neural network layers and Module base class
│   ├── functional.py       # Functional operations (activations, losses)
│   ├── optim.py            # Optimizers
│   └── schedulers.py       # Learning rate schedulers
│
├── 1_mlp_poc.ipynb         # MLP on MNIST - Optimizer comparison
├── 2_resnet.ipynb          # ResNet-18 on CIFAR-10
├── 3_attention.ipynb       # Multi-Head Attention for classification
├── LICENSE                 # Project license
└── README.md               # This file
```

## Features

### Neural Network Layers
- **Linear**: Fully connected layers
- **Conv2D**: 2D convolution with im2col implementation
- **BatchNorm2D**: Batch normalization for CNNs
- **LayerNorm**: Layer normalization for Transformers
- **MaxPool2D/AvgPool2D**: Pooling layers
- **Dropout**: Regularization
- **MultiHeadAttention**: Scaled dot-product attention
- **TransformerBlock**: Complete transformer layer

### Optimizers
- **SGD**: Stochastic Gradient Descent
- **Momentum**: SGD with momentum
- **Adagrad**: Adaptive learning rate based on gradients
- **RMSProp**: Root Mean Square Propagation
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with decoupled weight decay
- **Adadelta**: Fully implemented

## Demonstrations

### 1. MLP on MNIST (`1_mlp_poc.ipynb`)
Trains a 2-layer MLP on MNIST dataset comparing all 7 optimizers:
- **Dataset**: MNIST (10,000 samples subset)
- **Architecture**: 784 → 128 → 64 → 10
- **Goal**: Compare optimizer convergence and stability. This is the proof of concept for the library.

### 2. ResNet-18 on CIFAR-10 (`2_resnet.ipynb`)
Implements ResNet-18 from scratch with residual blocks:
- **Dataset**: CIFAR-10 (2,000 training samples)
- **Architecture**: 18-layer ResNet with skip connections
- **New Features added**: Conv2D, BatchNorm2D, Residual blocks
- **Goal**: Trained model with residual connections to demonstrate deep architectures.

### 3. Multi-Head Attention (`3_attention.ipynb`)
Implements Transformer-style attention for sequence classification:
- **Dataset**: Synthetic sentiment classification
- **Architecture**: Embedding → Attention Layers → Classifier
- **New Features added**: Multi-head attention, positional encoding, layer normalization
- **Goal**: Demonstrate attention mechanisms and transformer blocks.

## Installation & Usage

### Requirements
```bash
pip install numpy matplotlib scikit-learn seaborn
```

### 3. Convolutional Layers
Conv2D implemented using im2col for efficiency:
- Handles stride, padding, multiple channels
- Gradient computation for both inputs and filters

### 4. Multi-Head Attention
Full implementation of scaled dot-product attention:
- Query, Key, Value projections
- Multiple attention heads with parallel computation
- Compatible with Transformer architectures

## Performance Notes

Since this is a pure Python + NumPy implementation (no C++/CUDA):
- **Training is obviously slow** compared to PyTorch as it lacks GPU acceleration and low-level optimizations.
- CPU-only (no GPU support)

## Technical Decisions
- **im2col for Convolution**: Trade-off between memory and implementation simplicity
- **Scalar to Batched**: Extended from scalar autograd to full batched operations

## Learning Outcomes

This project demonstrates understanding of:
- Reverse-mode automatic differentiation
- Backpropagation through various operations
- Broadcasting and tensor shape manipulation
- Neural network layer implementations
- Optimizer algorithms and learning rate scheduling
- Modern architectures (ResNet, Transformers)

## Limitations

- No GPU support
- Slower than production frameworks
- Not suitable for large-scale training

## Contributors
- Angela Saade
- Aurelien Daudin
- Baptiste Arnold
- Khaled Mili