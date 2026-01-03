"""
mini_torch: A minimal PyTorch-like deep learning library.

This library implements automatic differentiation, neural network layers,
optimizers, and schedulers from scratch using only NumPy.
"""

from .tensor import Tensor
from . import functional as F
from . import nn
from . import optim
from . import schedulers

__version__ = '0.1.0'
__all__ = ['Tensor', 'F', 'nn', 'optim', 'schedulers']
