"""
Tensor class with automatic differentiation support for batched operations.
Supports scalar and multidimensional arrays with broadcasting.
"""
import numpy as np


class Tensor:
    """
    Stores a tensor (scalar or multidimensional array) and its gradient.
    Supports automatic differentiation with reverse-mode autodiff.
    """

    def __init__(self, data, _children=(), _op='', requires_grad=True):
        # Convert data to numpy array if not already
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = None if not requires_grad else np.zeros_like(self.data, dtype=np.float32)

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                # Handle broadcasting in backward pass
                grad = out.grad.copy()
                # Sum out added dims
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                # Sum over broadcasted dims
                for i, (dim, size) in enumerate(zip(grad.shape, self.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = out.grad.copy()
                # Sum out added dims
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                # Sum over broadcasted dims
                for i, (dim, size) in enumerate(zip(grad.shape, other.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, size) in enumerate(zip(grad.shape, self.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = self.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, (dim, size) in enumerate(zip(grad.shape, other.data.shape)):
                    if size == 1 and dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            if self.requires_grad:
                self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication with autograd support."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if self.requires_grad:
                # gradient w.r.t. self: grad @ other.T
                grad_self = out.grad @ other.data.swapaxes(-2, -1)
                # Handle dimension mismatch by summing over extra dims
                # We needed this for the attention model
                while grad_self.ndim > self.grad.ndim:
                    grad_self = grad_self.sum(axis=0)
                self.grad += grad_self
                
            if other.requires_grad:
                # gradient w.r.t. other: self.T @ grad
                grad_other = self.data.swapaxes(-2, -1) @ out.grad
                # Handle dimension mismatch by summing over batch dimensions
                # Same here, needed for attention model
                while grad_other.ndim > other.grad.ndim:
                    grad_other = grad_other.sum(axis=0)
                other.grad += grad_other

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum elements along given axis."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        # Add back the reduced dimension
                        if isinstance(axis, int):
                            grad = np.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, axis=ax)
                # Broadcast gradient to match input shape
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean of elements along given axis."""
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')
        n = self.data.size if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])

        def _backward():
            if self.requires_grad:
                grad = out.grad / n
                if axis is not None:
                    if not keepdims:
                        if isinstance(axis, int):
                            grad = np.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        """Max of elements along given axis."""
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), 'max')
        
        def _backward():
            if self.requires_grad:
                # Gradient flows only to maximum elements
                grad = out.grad
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                # Create mask for maximum elements
                if axis is None:
                    mask = (self.data == self.data.max()).astype(np.float32)
                else:
                    max_vals = np.expand_dims(out.data, axis=axis) if not keepdims else out.data
                    mask = (self.data == max_vals).astype(np.float32)
                
                # Distribute gradient to all maximum values
                grad_input = mask * np.broadcast_to(grad, self.data.shape)
                self.grad += grad_input

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Reshape tensor to new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """Transpose tensor dimensions."""
        if len(axes) == 0:
            # Default transpose: reverse all axes
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')

        def _backward():
            if self.requires_grad:
                if axes is None:
                    # Reverse transpose
                    self.grad += out.grad.transpose()
                else:
                    # Inverse permutation
                    inv_axes = list(range(len(axes)))
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    self.grad += out.grad.transpose(inv_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        """Transpose property for 2D tensors."""
        return self.transpose()

    def squeeze(self, axis=None):
        """Remove single-dimensional entries."""
        out = Tensor(self.data.squeeze(axis), (self,), 'squeeze')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def unsqueeze(self, axis):
        """Add a dimension of size 1."""
        out = Tensor(np.expand_dims(self.data, axis), (self,), 'unsqueeze')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        """ReLU activation function."""
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def build_topo(self, visited=None, topo=None):
        """Build topological ordering of computation graph."""
        if visited is None:
            visited = set()
        if topo is None:
            topo = []
            
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                child.build_topo(visited=visited, topo=topo)
            topo.append(self)
        return topo

    def backward(self):
        """Compute gradients using reverse-mode autodiff."""
        # Build topological order
        topo = self.build_topo()

        # Initialize gradient of output
        if self.grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float32)
        else:
            self.grad = np.ones_like(self.data, dtype=np.float32)

        # Apply chain rule in reverse order
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradients to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Tensor) else Tensor(other) ** -1)

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data}, grad={self.grad})"

    def __getitem__(self, idx):
        """Advanced indexing support."""
        out = Tensor(self.data[idx], (self,), 'getitem')
        
        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad
        
        out._backward = _backward
        return out

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Total number of elements."""
        return self.data.size

    def item(self):
        """Get scalar value (for single-element tensors)."""
        return self.data.item()

    def numpy(self):
        """Return data as numpy array."""
        return self.data
