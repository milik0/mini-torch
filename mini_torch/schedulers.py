"""
Learning rate schedulers for mini_torch optimizers.
"""
import numpy as np


class LRScheduler:
    """
    Base class for learning rate schedulers.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        initial_lr: Initial learning rate
    """
    
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.iteration = 0
    
    def update_lr(self, metrics=None):
        """
        Update the learning rate.
        Should be overridden by subclasses.
        
        Args:
            metrics: Optional metric value for scheduling
        """
        self.optimizer.learning_rate = self.lr
    
    def step(self, metrics=None):
        """
        Perform a scheduling step.
        
        Args:
            metrics: Optional metric value for scheduling
        """
        self.iteration += 1
        self.update_lr(metrics)


class LRSchedulerOnPlateau(LRScheduler):
    """
    Reduce learning rate when a metric has stopped improving.
    
    Args:
        optimizer: Optimizer to schedule
        initial_lr: Initial learning rate
        patience: Number of epochs with no improvement before reducing LR (default: 10)
        factor: Factor by which to reduce learning rate (default: 0.1)
        min_lr: Minimum learning rate (default: 1e-6)
        mode: 'min' for minimization, 'max' for maximization (default: 'min')
        threshold: Threshold for measuring improvement (default: 1e-4)
    """
    
    def __init__(self, optimizer, initial_lr, patience=10, factor=0.1, 
                 min_lr=1e-6, mode='min', threshold=1e-4):
        super().__init__(optimizer, initial_lr)
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        
        # Initialize best metric based on mode
        if mode == 'min':
            self.best_metric = float('inf')
        elif mode == 'max':
            self.best_metric = float('-inf')
        else:
            raise ValueError(f"Mode {mode} is invalid. Use 'min' or 'max'.")
        
        self.num_bad_epochs = 0
    
    def update_lr(self, metric):
        """
        Update learning rate based on metric performance.
        
        Args:
            metric: Current metric value to monitor
        """
        if metric is None:
            return
        
        # Check if metric has improved
        if self.mode == 'min':
            improved = metric < (self.best_metric - self.threshold)
        else:  # mode == 'max'
            improved = metric > (self.best_metric + self.threshold)
        
        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce learning rate if no improvement for 'patience' epochs
        if self.num_bad_epochs >= self.patience:
            old_lr = self.lr
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.optimizer.learning_rate = self.lr
            self.num_bad_epochs = 0  # Reset counter after reduction
            
            if self.lr != old_lr:
                print(f"Reducing learning rate from {old_lr:.6f} to {self.lr:.6f}")


class StepLR(LRScheduler):
    """
    Decay learning rate by factor every step_size epochs.
    
    Args:
        optimizer: Optimizer to schedule
        initial_lr: Initial learning rate
        step_size: Period of learning rate decay (in epochs)
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
    """
    
    def __init__(self, optimizer, initial_lr, step_size, gamma=0.1):
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def update_lr(self, metrics=None):
        """Update learning rate every step_size iterations"""
        if self.iteration % self.step_size == 0 and self.iteration > 0:
            self.lr = self.lr * self.gamma
            self.optimizer.learning_rate = self.lr


class ExponentialLR(LRScheduler):
    """
    Decay learning rate exponentially.
    
    Args:
        optimizer: Optimizer to schedule
        initial_lr: Initial learning rate
        gamma: Multiplicative factor of learning rate decay per epoch
    """
    
    def __init__(self, optimizer, initial_lr, gamma):
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma
    
    def update_lr(self, metrics=None):
        """Exponentially decay learning rate"""
        self.lr = self.initial_lr * (self.gamma ** self.iteration)
        self.optimizer.learning_rate = self.lr


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    Args:
        optimizer: Optimizer to schedule
        initial_lr: Initial learning rate
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate (default: 0)
    """
    
    def __init__(self, optimizer, initial_lr, T_max, eta_min=0):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def update_lr(self, metrics=None):
        """Update learning rate using cosine annealing"""
        self.lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                  (1 + np.cos(np.pi * self.iteration / self.T_max)) / 2
        self.optimizer.learning_rate = self.lr
