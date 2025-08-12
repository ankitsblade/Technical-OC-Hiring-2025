class SGD:
    def __init__(self, lr=0.1, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def step(self, layers):
        for layer in layers:
            # fetch param/grad pairs from layer (may be empty iterator)
            fn = getattr(layer, "params_and_grads", None)
            if fn is None:
                continue
            for p, g in fn():
                if self.wd != 0.0:
                    g = g + self.wd * p
                p -= self.lr * g

    def zero_grad(self, layers):
        for layer in layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
