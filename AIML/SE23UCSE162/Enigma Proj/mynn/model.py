import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_fn = loss
        self.opt = optimizer
        self.compiled = True

    def _forward(self, x, training=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def _backward(self, grad_out):
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X, y_onehot, epochs=50, batch_size=64, verbose=1, X_val=None, y_val=None, metric_fn=None):
        assert self.compiled, "Call compile(loss, optimizer) first."
        n = X.shape[0]
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs+1):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y_onehot[idx]

            self.opt.zero_grad(self.layers)
            for start in range(0, n, batch_size):
                end = start + batch_size
                xb = Xs[start:end]
                yb = ys[start:end]

                logits = self._forward(xb, training=True)
                loss = self.loss_fn.forward(logits, yb)
                grad_logits = self.loss_fn.backward()
                self._backward(grad_logits)
                self.opt.step(self.layers)
                self.opt.zero_grad(self.layers)

            log = f"Epoch {epoch:03d} | loss={loss:.4f}"
            if metric_fn is not None and X_val is not None and y_val is not None:
                logits_val = self._forward(X_val, training=False)
                acc = metric_fn(logits_val, y_val)
                log += f" | val_acc={acc*100:.2f}%"
            if verbose:
                print(log)

    def predict(self, X):
        logits = self._forward(X, training=False)
        return logits

    def evaluate(self, X, y, metric_fn):
        logits = self._forward(X, training=False)
        return metric_fn(logits, y)
