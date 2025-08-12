import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, y_onehot):
        # combine softmax + CE in one stable op
        z = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_probs = z - logsumexp
        loss = - (y_onehot * log_probs).sum(axis=1).mean()
        self.probs = np.exp(log_probs)
        self.y = y_onehot
        return loss

    def backward(self):
        # dL/dlogits = (softmax - y) / batch
        batch = self.y.shape[0]
        return (self.probs - self.y) / batch
