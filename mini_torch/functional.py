"""
Functional operations for mini_torch: activations, losses, and other operations.
"""
import numpy as np
from .tensor import Tensor


# ============================================================================
# Activation Functions
# ============================================================================

def relu(x):
    """ReLU activation: max(0, x)"""
    if isinstance(x, Tensor):
        return x.relu()
    return Tensor(np.maximum(0, x.data if isinstance(x, Tensor) else x))


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    # For numerical stability
    out_data = np.where(
        x.data >= 0,
        1 / (1 + np.exp(-x.data)),
        np.exp(x.data) / (1 + np.exp(x.data))
    )
    out = Tensor(out_data, (x,), 'sigmoid')

    def _backward():
        if x.requires_grad:
            x.grad += out.data * (1 - out.data) * out.grad

    out._backward = _backward
    return out


def tanh(x):
    """Hyperbolic tangent activation"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.tanh(x.data), (x,), 'tanh')

    def _backward():
        if x.requires_grad:
            x.grad += (1 - out.data ** 2) * out.grad

    out._backward = _backward
    return out


def softmax(x, axis=-1):
    """
    Softmax activation along specified axis.
    Numerically stable implementation.
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    # Numerical stability: subtract max
    x_max = x.data.max(axis=axis, keepdims=True)
    exp_values = np.exp(x.data - x_max)
    out_data = exp_values / exp_values.sum(axis=axis, keepdims=True)
    
    out = Tensor(out_data, (x,), 'softmax')

    def _backward():
        if x.requires_grad:
            # Jacobian of softmax: diag(s) - s @ s.T
            # For batched inputs, we compute this efficiently
            grad = out.grad
            sum_grad = (grad * out.data).sum(axis=axis, keepdims=True)
            x.grad += out.data * (grad - sum_grad)

    out._backward = _backward
    return out


def log_softmax(x, axis=-1):
    """Log-softmax for numerical stability in cross-entropy loss."""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    x_max = x.data.max(axis=axis, keepdims=True)
    log_sum_exp = np.log(np.exp(x.data - x_max).sum(axis=axis, keepdims=True))
    out_data = x.data - x_max - log_sum_exp
    
    out = Tensor(out_data, (x,), 'log_softmax')

    def _backward():
        if x.requires_grad:
            grad = out.grad
            sum_grad = grad.sum(axis=axis, keepdims=True)
            x.grad += grad - np.exp(out.data) * sum_grad

    out._backward = _backward
    return out


# ============================================================================
# Mathematical Functions
# ============================================================================

def exp(x):
    """Exponential function"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.exp(x.data), (x,), 'exp')

    def _backward():
        if x.requires_grad:
            x.grad += out.data * out.grad

    out._backward = _backward
    return out


def log(x):
    """Natural logarithm"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.log(x.data), (x,), 'log')

    def _backward():
        if x.requires_grad:
            x.grad += (1 / x.data) * out.grad

    out._backward = _backward
    return out


def sin(x):
    """Sine function"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.sin(x.data), (x,), 'sin')

    def _backward():
        if x.requires_grad:
            x.grad += np.cos(x.data) * out.grad

    out._backward = _backward
    return out


def cos(x):
    """Cosine function"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.cos(x.data), (x,), 'cos')

    def _backward():
        if x.requires_grad:
            x.grad += -np.sin(x.data) * out.grad

    out._backward = _backward
    return out


def sqrt(x):
    """Square root function"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.sqrt(x.data), (x,), 'sqrt')

    def _backward():
        if x.requires_grad:
            x.grad += (0.5 / np.sqrt(x.data)) * out.grad

    out._backward = _backward
    return out


# ============================================================================
# Loss Functions
# ============================================================================

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss"""
    if not isinstance(y_pred, Tensor):
        y_pred = Tensor(y_pred)
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    
    diff = y_pred - y_true
    return (diff * diff).mean()


def binary_cross_entropy(y_pred, y_true):
    """Binary cross-entropy loss"""
    if not isinstance(y_pred, Tensor):
        y_pred = Tensor(y_pred)
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true, requires_grad=False)
    
    # Clamp predictions for numerical stability
    eps = 1e-8
    y_pred_clamped = Tensor(np.clip(y_pred.data, eps, 1 - eps), (y_pred,), 'clamp')
    
    def _backward():
        if y_pred.requires_grad:
            y_pred.grad += y_pred_clamped.grad
    y_pred_clamped._backward = _backward
    
    loss = -(y_true * log(y_pred_clamped) + (1 - y_true) * log(1 - y_pred_clamped))
    return loss.mean()


def cross_entropy(logits, targets):
    """
    Cross-entropy loss for multi-class classification.
    
    Args:
        logits: Tensor of shape (batch_size, num_classes) - raw scores
        targets: Tensor of shape (batch_size,) - class indices
    """
    if not isinstance(logits, Tensor):
        logits = Tensor(logits)
    if not isinstance(targets, Tensor):
        targets = Tensor(targets, requires_grad=False)
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Compute log softmax for numerical stability
    log_probs = log_softmax(logits, axis=1)
    
    # Select log probabilities for target classes
    targets_int = targets.data.astype(np.int32)
    selected_log_probs = log_probs.data[np.arange(batch_size), targets_int]
    
    # Negative log likelihood
    out = Tensor(-selected_log_probs.mean(), (log_probs,), 'cross_entropy')
    
    def _backward():
        if log_probs.requires_grad:
            # Gradient of NLL w.r.t. log probabilities
            grad = np.zeros_like(log_probs.data)
            grad[np.arange(batch_size), targets_int] = -1.0 / batch_size
            log_probs.grad += grad * out.grad
    
    out._backward = _backward
    return out


# ============================================================================
# Utility Functions
# ============================================================================

def concat(tensors, axis=0):
    """Concatenate tensors along specified axis"""
    if not all(isinstance(t, Tensor) for t in tensors):
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), 'concat')
    
    def _backward():
        grad_split = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, grad_split):
            if t.requires_grad:
                t.grad += g
    
    out._backward = _backward
    return out


def stack(tensors, axis=0):
    """Stack tensors along new dimension"""
    if not all(isinstance(t, Tensor) for t in tensors):
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), 'stack')
    
    def _backward():
        for i, t in enumerate(tensors):
            if t.requires_grad:
                # Take gradient for this element from stacked dimension
                grad = np.take(out.grad, i, axis=axis)
                t.grad += grad
    
    out._backward = _backward
    return out


def pad(x, pad_width, mode='constant', constant_value=0):
    """Pad tensor with specified mode"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.pad(x.data, pad_width, mode=mode, constant_values=constant_value), (x,), 'pad')
    
    def _backward():
        if x.requires_grad:
            # Remove padding from gradient
            slices = []
            for (left, right) in pad_width:
                if right == 0:
                    slices.append(slice(left, None))
                else:
                    slices.append(slice(left, -right))
            x.grad += out.grad[tuple(slices)]
    
    out._backward = _backward
    return out


def dropout(x, p=0.5, training=True):
    """
    Dropout regularization.
    
    Args:
        x: Input tensor
        p: Probability of dropping a unit (0 to 1)
        training: Whether in training mode
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not training or p == 0:
        return x
    
    # Generate dropout mask
    mask = (np.random.rand(*x.shape) > p).astype(np.float32)
    # Scale by 1/(1-p) to maintain expected value
    scale = 1.0 / (1.0 - p)
    
    out = Tensor(x.data * mask * scale, (x,), 'dropout')
    
    def _backward():
        if x.requires_grad:
            x.grad += out.grad * mask * scale
    
    out._backward = _backward
    return out
