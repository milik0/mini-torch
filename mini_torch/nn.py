"""
Neural network layers and modules for mini_torch.
Includes Module base class, Linear, Conv2D, BatchNorm, LayerNorm, Pooling, etc.
"""
import numpy as np
from .tensor import Tensor
from . import functional as F


class Module:
    """
    Base class for all neural network modules.
    Similar to PyTorch's nn.Module.
    """
    
    def __init__(self):
        self._training = True
        self._parameters = []
    
    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        """Make module callable"""
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """Return list of all parameters in this module"""
        return self._parameters
    
    def train(self):
        """Set module to training mode"""
        self._training = True
        return self
    
    def eval(self):
        """Set module to evaluation mode"""
        self._training = False
        return self
    
    def zero_grad(self):
        """Zero out all parameter gradients"""
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """
    Fully connected (dense) layer: y = xW + b
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with Xavier/Glorot initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * std)
        
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None
        
        self._parameters = [self.weight]
        if self.bias is not None:
            self._parameters.append(self.bias)
    
    def forward(self, x):
        """
        Forward pass: y = xW + b
        
        Args:
            x: Input tensor of shape (batch_size, in_features) or (in_features,)
        """
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2D(Module):
    """
    2D Convolution layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output filters
        kernel_size: Size of convolution kernel (int or tuple)
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        bias: Whether to include bias term
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        # Initialize weights: (out_channels, in_channels, kH, kW)
        k_h, k_w = self.kernel_size
        std = np.sqrt(2.0 / (in_channels * k_h * k_w))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, k_h, k_w).astype(np.float32) * std
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None
        
        self._parameters = [self.weight]
        if self.bias is not None:
            self._parameters.append(self.bias)
    
    def forward(self, x):
        """
        Forward pass using im2col method for efficiency.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        batch_size, in_channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Calculate output dimensions
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1
        
        # Apply padding if needed
        if p_h > 0 or p_w > 0:
            x_padded = F.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')
        else:
            x_padded = x
        
        # im2col: convert image patches to columns
        col = self._im2col(x_padded.data, k_h, k_w, s_h, s_w, out_h, out_w)
        
        # Reshape weights: (out_channels, in_channels * kH * kW)
        weight_col = self.weight.data.reshape(self.out_channels, -1)
        
        # Matrix multiplication: (batch, out_h*out_w, in*kH*kW) @ (in*kH*kW, out_channels).T
        out_data = col @ weight_col.T  # (batch, out_h*out_w, out_channels)
        
        # Reshape to (batch, out_channels, out_h, out_w)
        out_data = out_data.reshape(batch_size, out_h, out_w, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        
        out = Tensor(out_data, (x, self.weight), 'conv2d')
        
        # Capture variables in closure for backward pass
        conv_cache = {
            'x_padded': x_padded,
            'col': col,
            'weight_col': weight_col,
            'out_h': out_h,
            'out_w': out_w,
            'batch_size': batch_size
        }
        
        def _backward():
            # Use cached values from closure
            cache = conv_cache
            
            # Gradient w.r.t. output: (batch, out_channels, out_h, out_w)
            dout = out.grad
            
            # Reshape: NCHW -> NHWC -> (batch*out_h*out_w, out_channels)
            dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
            
            # Gradient w.r.t. weight
            if self.weight.requires_grad:
                # dW = col.T @ dout_reshaped
                dweight = cache['col'].T @ dout_reshaped  # (in*kH*kW, out_channels)
                dweight = dweight.T.reshape(self.weight.shape)  # (out_channels, in, kH, kW)
                self.weight.grad += dweight
            
            # Gradient w.r.t. input
            if x.requires_grad:
                # dx_col = dout_reshaped @ weight_col
                dx_col = dout_reshaped @ cache['weight_col']  # (batch*out_h*out_w, in*kH*kW)
                
                # col2im: convert columns back to image
                dx_padded = self._col2im(
                    dx_col, batch_size, in_channels, 
                    in_h + 2 * p_h, in_w + 2 * p_w,
                    k_h, k_w, s_h, s_w, cache['out_h'], cache['out_w']
                )
                
                # Remove padding
                if p_h > 0 or p_w > 0:
                    dx = dx_padded[:, :, p_h:-p_h, p_w:-p_w]
                else:
                    dx = dx_padded
                
                x.grad += dx
            
            # Gradient w.r.t. bias
            if self.bias is not None and self.bias.requires_grad:
                dbias = dout.sum(axis=(0, 2, 3))  # Sum over batch, height, width
                self.bias.grad += dbias
        
        out._backward = _backward
        
        # Add bias if present
        if self.bias is not None:
            # Reshape bias for broadcasting: (1, out_channels, 1, 1)
            bias_reshaped = self.bias.reshape(1, self.out_channels, 1, 1)
            out = out + bias_reshaped
        
        return out
    
    def _im2col(self, x, k_h, k_w, s_h, s_w, out_h, out_w):
        """Convert image patches to columns for efficient convolution"""
        batch_size, in_channels, h, w = x.shape
        
        col = np.zeros((batch_size * out_h * out_w, in_channels * k_h * k_w), dtype=np.float32)
        
        idx = 0
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * s_h
                    w_start = j * s_w
                    patch = x[b, :, h_start:h_start+k_h, w_start:w_start+k_w]
                    col[idx] = patch.reshape(-1)
                    idx += 1
        
        return col
    
    def _col2im(self, col, batch_size, in_channels, h, w, k_h, k_w, s_h, s_w, out_h, out_w):
        """Convert columns back to image for backward pass"""
        x = np.zeros((batch_size, in_channels, h, w), dtype=np.float32)
        
        idx = 0
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * s_h
                    w_start = j * s_w
                    patch = col[idx].reshape(in_channels, k_h, k_w)
                    x[b, :, h_start:h_start+k_h, w_start:w_start+k_w] += patch
                    idx += 1
        
        return x


class BatchNorm2D(Module):
    """
    Batch Normalization for 2D inputs (images).
    
    Args:
        num_features: Number of channels (C from input shape (N, C, H, W))
        eps: Small constant for numerical stability
        momentum: Momentum for running mean/variance
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32))  # Scale
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32))  # Shift
        
        # Running statistics (not trainable)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        self._parameters = [self.gamma, self.beta]
    
    def forward(self, x):
        """
        Forward pass for BatchNorm2D.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        batch_size, num_features, h, w = x.shape
        
        if self._training:
            # Calculate batch statistics
            # Mean and var over batch, height, width dimensions
            mean = x.data.mean(axis=(0, 2, 3), keepdims=False)  # (num_features,)
            var = x.data.var(axis=(0, 2, 3), keepdims=False)   # (num_features,)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize: reshape for broadcasting (1, C, 1, 1)
        mean_reshaped = mean.reshape(1, num_features, 1, 1)
        var_reshaped = var.reshape(1, num_features, 1, 1)
        
        # Standardize
        x_centered = x - Tensor(mean_reshaped, requires_grad=False)
        std = Tensor(np.sqrt(var_reshaped + self.eps), requires_grad=False)
        x_norm = x_centered / std
        
        # Scale and shift
        gamma_reshaped = self.gamma.reshape(1, num_features, 1, 1)
        beta_reshaped = self.beta.reshape(1, num_features, 1, 1)
        
        out = gamma_reshaped * x_norm + beta_reshaped
        
        return out


class LayerNorm(Module):
    """
    Layer Normalization.
    
    Args:
        normalized_shape: Shape of features to normalize over (tuple or int)
        eps: Small constant for numerical stability
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32))
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32))
        
        self._parameters = [self.gamma, self.beta]
    
    def forward(self, x):
        """
        Forward pass for LayerNorm.
        
        Args:
            x: Input tensor
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Calculate which axes to normalize over (last len(normalized_shape) axes)
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        # Calculate mean and variance
        mean = x.mean(axis=axes, keepdims=True)
        var_data = ((x - mean).data ** 2).mean(axis=axes, keepdims=True)
        
        # Standardize
        x_norm = (x - mean) / Tensor(np.sqrt(var_data + self.eps), requires_grad=False)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        return out


class MaxPool2D(Module):
    """
    2D Max Pooling layer.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: same as kernel_size)
        padding: Padding added to input (default: 0)
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x):
        """
        Forward pass for MaxPool2D.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        batch_size, channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Apply padding if needed
        if p_h > 0 or p_w > 0:
            x_padded = F.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant', constant_value=-np.inf)
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1
        
        # Perform max pooling
        out_data = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
        max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        patch = x_padded.data[b, c, h_start:h_start+k_h, w_start:w_start+k_w]
                        max_val = patch.max()
                        out_data[b, c, i, j] = max_val
                        
                        # Store indices of max value for backward pass
                        max_idx = np.unravel_index(patch.argmax(), patch.shape)
                        max_indices[b, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        out = Tensor(out_data, (x,), 'maxpool2d')
        out._max_indices = max_indices
        out._input_shape = x.shape
        
        def _backward():
            if x.requires_grad:
                dx_padded = np.zeros(x_padded.shape, dtype=np.float32)
                
                for b in range(batch_size):
                    for c in range(channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_idx, w_idx = max_indices[b, c, i, j]
                                dx_padded[b, c, h_idx, w_idx] += out.grad[b, c, i, j]
                
                # Remove padding
                if p_h > 0 or p_w > 0:
                    dx = dx_padded[:, :, p_h:-p_h, p_w:-p_w]
                else:
                    dx = dx_padded
                
                x.grad += dx
        
        out._backward = _backward
        return out


class AvgPool2D(Module):
    """
    2D Average Pooling layer.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: same as kernel_size)
        padding: Padding added to input (default: 0)
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x):
        """Forward pass for AvgPool2D"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        batch_size, channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Apply padding if needed
        if p_h > 0 or p_w > 0:
            x_padded = F.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1
        
        # Perform average pooling
        out_data = np.zeros((batch_size, channels, out_h, out_w), dtype=np.float32)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        patch = x_padded.data[b, c, h_start:h_start+k_h, w_start:w_start+k_w]
                        out_data[b, c, i, j] = patch.mean()
        
        out = Tensor(out_data, (x,), 'avgpool2d')
        
        def _backward():
            if x.requires_grad:
                dx_padded = np.zeros(x_padded.shape, dtype=np.float32)
                grad_scale = 1.0 / (k_h * k_w)
                
                for b in range(batch_size):
                    for c in range(channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_start = i * s_h
                                w_start = j * s_w
                                dx_padded[b, c, h_start:h_start+k_h, w_start:w_start+k_w] += \
                                    out.grad[b, c, i, j] * grad_scale
                
                # Remove padding
                if p_h > 0 or p_w > 0:
                    dx = dx_padded[:, :, p_h:-p_h, p_w:-p_w]
                else:
                    dx = dx_padded
                
                x.grad += dx
        
        out._backward = _backward
        return out


class Dropout(Module):
    """
    Dropout layer for regularization.
    
    Args:
        p: Probability of dropping a unit (between 0 and 1)
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        """Forward pass for Dropout"""
        return F.dropout(x, p=self.p, training=self._training)


class Flatten(Module):
    """Flatten all dimensions except batch dimension"""
    
    def forward(self, x):
        """Flatten input tensor"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class Sequential(Module):
    """
    Sequential container for modules.
    Passes input through each module in order.
    """
    
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = list(modules)
        
        # Collect all parameters from submodules
        self._parameters = []
        for module in self.modules_list:
            if isinstance(module, Module):
                self._parameters.extend(module.parameters())
    
    def forward(self, x):
        """Forward pass through all modules"""
        for module in self.modules_list:
            x = module(x)
        return x
    
    def train(self):
        """Set all modules to training mode"""
        self._training = True
        for module in self.modules_list:
            if isinstance(module, Module):
                module.train()
        return self
    
    def eval(self):
        """Set all modules to evaluation mode"""
        self._training = False
        for module in self.modules_list:
            if isinstance(module, Module):
                module.eval()
        return self


class MultiHeadAttention(Module):
    """
    Multi-Head Attention mechanism.
    
    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        
        # Query, Key, Value projection matrices
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = Linear(embed_dim, embed_dim)
        
        # Collect parameters
        self._parameters = (self.W_q.parameters() + self.W_k.parameters() + 
                           self.W_v.parameters() + self.W_o.parameters())
    
    def forward(self, query, key=None, value=None, mask=None):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor (if None, uses query)
            value: Value tensor (if None, uses query)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, embed_dim)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads: (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # Attention scores: Q @ K^T / sqrt(head_dim)
        scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len, seq_len)
        scores = scores * (1.0 / np.sqrt(self.head_dim))
        
        # Apply mask if provided
        if mask is not None:
            # Mask should be (seq_len, seq_len) or (batch, seq_len, seq_len)
            scores = scores + Tensor(mask * -1e9, requires_grad=False)
        
        # Softmax over last dimension
        attn_weights = F.softmax(scores, axis=-1)
        
        # Apply dropout if training
        if self._training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=True)
        
        # Attention output: attn_weights @ V
        attn_output = attn_weights @ V  # (batch, num_heads, seq_len, head_dim)
        
        # Transpose back and concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class TransformerBlock(Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    
    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ff = Sequential(
            Linear(embed_dim, ff_dim),
            # ReLU is applied manually
            Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # Collect parameters
        self._parameters = (self.attention.parameters() + self.ff.parameters() + 
                           self.norm1.parameters() + self.norm2.parameters())
    
    def forward(self, x, mask=None):
        """
        Forward pass for Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        """
        # Multi-head attention with residual connection
        attn_output = self.attention(x, mask=mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ff_output = self.ff.modules_list[0](x)  # First linear layer
        ff_output = ff_output.relu()  # ReLU activation
        ff_output = self.ff.modules_list[1](ff_output)  # Second linear layer
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)
        
        return x
