import numpy as np
from .layers import Layer

class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = (x > 0).astype(x.dtype)
        return np.maximum(0, x)
    def backward(self, grad_out):
        return grad_out * self.mask

class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out
    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)

class Softmax(Layer):
    # Usually used as the final layer; gradient handled in loss for stability,
    # but we keep this for completeness if needed.
    def forward(self, x, training=True):
        x = x - x.max(axis=1, keepdims=True)
        exp = np.exp(x)
        self.out = exp / exp.sum(axis=1, keepdims=True)
        return self.out
    def backward(self, grad_out):
        # Not commonly used directly with CE; provided for completeness.
        # Jacobian-vector product for softmax. Slower but correct.
        batch = grad_out.shape[0]
        grad_in = np.empty_like(grad_out)
        for i in range(batch):
            s = self.out[i].reshape(-1,1)
            J = np.diagflat(s) - s @ s.T
            grad_in[i] = J @ grad_out[i]
        return grad_in
