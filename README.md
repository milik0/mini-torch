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

### 1. MLP on MNIST ([1_mlp_poc.ipynb](./1_mlp_poc.ipynb))
Trains a 2-layer MLP on MNIST dataset comparing all 7 optimizers:
- **Dataset**: MNIST (10,000 samples subset)
- **Architecture**: 784 → 128 → 64 → 10
- **Goal**: Compare optimizer convergence and stability. This is the proof of concept for the library.

### 2. ResNet-18 on CIFAR-10 ([2_resnet.ipynb](./2_resnet.ipynb))
Implements ResNet-18 from scratch with residual blocks:
- **Dataset**: CIFAR-10 (2,000 training samples)
- **Architecture**: 18-layer ResNet with skip connections
- **New Features added**: Conv2D, BatchNorm2D, Residual blocks
- **Goal**: Trained model with residual connections to demonstrate deep architectures.

### 3. Multi-Head Attention for Sequence Classification ([3_attention.ipynb](./3_attention.ipynb))
Implements Transformer-style attention for sequence classification:
- **Dataset**: Synthetic sentiment classification
- **Architecture**: Embedding → Attention Layers → Classifier
- **New Features added**: Multi-head attention, positional encoding, layer normalization
- **Goal**: Demonstrate attention mechanisms and transformer blocks.

## Implementation Details for new Features

### 1. Convolutional Layers
Conv2D implemented using im2col for efficiency:
- Handles stride, padding, multiple channels
- Gradient computation for both inputs and filters

### 2. Multi-Head Attention
Full implementation of scaled dot-product attention:
- Query, Key, Value projections
- Multiple attention heads with parallel computation
- Compatible with Transformer architectures

## Technical Decisions
- **im2col for Convolution**: Trade-off between memory and implementation simplicity
- **Scalar to Batched**: Extended from scalar autograd to full batched operations

## Performance Notes

Since this is a pure Python + NumPy implementation (no C++/CUDA):
- **Training is obviously slow** compared to PyTorch as it lacks GPU acceleration and low-level optimizations.
- CPU-only (no GPU support)

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

## Installation & Usage

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable Python dependency management. If you don't have uv installed, please install it first. Then follow the steps below:

1. **Install dependencies and library**:
   ```bash
   uv sync
   ```
   
   This command will:
   - Create a virtual environment in `.venv/`
   - Install all required dependencies (numpy, matplotlib, scikit-learn, pandas, jupyter, notebook, torchvision)
   - Install the `mini_torch` library in editable mode

### Running the Notebooks

Start Jupyter Notebook in the virtual environment.
Then open any of the demonstration notebooks:
- `1_mlp_poc.ipynb` - MLP on MNIST with optimizer comparison
- `2_resnet.ipynb` - ResNet-18 on CIFAR-10
- `3_attention.ipynb` - Multi-Head Attention for classification


## Contributors
- Angela Saade
- Aurelien Daudin
- Baptiste Arnold
- Khaled Mili