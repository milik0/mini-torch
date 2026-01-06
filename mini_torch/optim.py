"""
Optimizers for mini_torch.
Includes SGD, Momentum, Adagrad, RMSProp, Adam, AdamW, and Adadelta.
"""
import numpy as np


class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params):
        self.params = list(params)
    
    def step(self):
        """Update parameters - to be implemented by subclasses"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out all parameter gradients"""
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        params: Parameters to optimize
        learning_rate (or lr): Learning rate (step size)
    """
    
    def __init__(self, params, learning_rate=0.01, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
    
    def step(self):
        """Update parameters using SGD"""
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                param.data -= self.learning_rate * param.grad


class Momentum(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        """Update parameters using PyTorch-style Momentum"""
        for param, velocity in zip(self.params, self.velocity):
            if param.requires_grad and param.grad is not None:
                # Update velocity (accumulation of gradients)
                velocity[:] = self.momentum * velocity + param.grad
                # Remark: The velocity variable stores a history of gradients scaled by the previous (larger) learning rate.
                # When the scheduler drops the learning rate (e.g., from 0.001 to 0.0005), the "inertia" from the previous velocity
                # is still huge. The optimizer effectively ignores the new, smaller learning rate for several steps,
                # causing the model to overshoot and the loss to explode
                
                # Apply update scaled by learning rate
                param.data -= self.learning_rate * velocity


class Adagrad(Optimizer):
    """
    Adagrad (Adaptive Gradient) optimizer.
    Adapts learning rate based on cumulative sum of squared gradients.
    
    Args:
        params: Parameters to optimize
        learning_rate (or lr): Learning rate
        eps: Small constant for numerical stability
    """
    
    def __init__(self, params, learning_rate=0.01, eps=1e-8, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        """Update parameters using Adagrad"""
        for param, cache in zip(self.params, self.cache):
            if param.requires_grad and param.grad is not None:
                # Accumulate squared gradients
                cache[:] = cache + param.grad ** 2
                
                # Update parameter with adaptive learning rate
                param.data -= self.learning_rate * param.grad / (np.sqrt(cache) + self.eps)


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation) optimizer.
    Uses exponential moving average of squared gradients.
    
    Args:
        params: Parameters to optimize
        learning_rate (or lr): Learning rate
        decay: Decay rate for moving average (default: 0.9)
        eps: Small constant for numerical stability
    """
    
    def __init__(self, params, learning_rate=0.01, decay=0.9, eps=1e-8, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
        self.decay = decay
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        """Update parameters using RMSProp"""
        for param, cache in zip(self.params, self.cache):
            if param.requires_grad and param.grad is not None:
                # Update cache with exponential moving average
                cache[:] = self.decay * cache + (1 - self.decay) * (param.grad ** 2)
                
                # Update parameter with adaptive learning rate
                param.data -= self.learning_rate * param.grad / (np.sqrt(cache) + self.eps)


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    Combines momentum and RMSProp with bias correction.
    
    Args:
        params: Parameters to optimize
        learning_rate (or lr): Learning rate
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability
    """
    
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        """Update parameters using Adam"""
        self.t += 1
        
        for param, m, v in zip(self.params, self.m, self.v):
            if param.requires_grad and param.grad is not None:
                # Update biased first moment estimate
                m[:] = self.beta1 * m + (1 - self.beta1) * param.grad
                
                # Update biased second moment estimate
                v[:] = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second moment estimate
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # Update parameter
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW (Adam with Weight Decay) optimizer.
    Adam with decoupled weight decay regularization.
    
    Args:
        params: Parameters to optimize
        learning_rate (or lr): Learning rate
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, lr=None):
        super().__init__(params)
        self.learning_rate = lr if lr is not None else learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
    
    def step(self):
        """Update parameters using AdamW"""
        self.t += 1
        
        for param, m, v in zip(self.params, self.m, self.v):
            if param.requires_grad and param.grad is not None:
                # Apply weight decay (decoupled from gradient-based update)
                param.data -= self.learning_rate * self.weight_decay * param.data
                
                # Update biased first moment estimate
                m[:] = self.beta1 * m + (1 - self.beta1) * param.grad
                
                # Update biased second moment estimate
                v[:] = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)
                
                # Compute bias-corrected moments
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # Update parameter
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class Adadelta(Optimizer):
    """
    Adadelta optimizer.
    Adaptive learning rate method that uses exponential moving averages
    of squared gradients and squared parameter updates.
    
    Args:
        params: Parameters to optimize
        rho: Decay rate for moving averages (default: 0.95)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, params, rho=0.95, eps=1e-8):
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]  # Accumulated squared gradients
        self.delta = [np.zeros_like(p.data) for p in params]  # Accumulated squared updates
    
    def step(self):
        """
        Update parameters using Adadelta algorithm.
        
        The Adadelta algorithm adapts learning rates based on a moving window
        of gradient updates, without requiring a manual learning rate.
        
        Algorithm:
        1. Accumulate gradient: cache = rho * cache + (1 - rho) * grad^2
        2. Compute update: update = -grad * sqrt(delta + eps) / sqrt(cache + eps)
        3. Accumulate update: delta = rho * delta + (1 - rho) * update^2
        4. Apply update: param = param + update
        """
        for param, cache, delta in zip(self.params, self.cache, self.delta):
            if param.requires_grad and param.grad is not None:
                # Update cache (exponential moving average of squared gradients)
                cache[:] = self.rho * cache + (1 - self.rho) * (param.grad ** 2)
                
                # Compute parameter update using RMS of delta and cache
                update = -param.grad * np.sqrt(delta + self.eps) / np.sqrt(cache + self.eps)
                
                # Update delta (exponential moving average of squared updates)
                delta[:] = self.rho * delta + (1 - self.rho) * (update ** 2)
                
                # Apply parameter update
                param.data += update
