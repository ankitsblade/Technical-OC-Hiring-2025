import numpy as np

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad_out): raise NotImplementedError
    def params_and_grads(self):
        yield from ()
    def zero_grad(self):
        return

class Dense(Layer):
    def __init__(self, in_features, out_features, rng=None):
        self.rng = rng or np.random.default_rng(0)
        self.W = self.rng.normal(0, 0.1, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        # grad_out shape: (batch, out_features)
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0, keepdims=True)
        grad_in = grad_out @ self.W.T
        return grad_in

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db

    def zero_grad(self):
        self.dW.fill(0.0); self.db.fill(0.0)
