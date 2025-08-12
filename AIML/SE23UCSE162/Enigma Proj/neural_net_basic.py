# neural_net_basic.py
# Minimal 2-layer classifier using ONLY numpy (and optionally pandas if you swap dataset loading).
# Trains on a synthetic 3-class "blobs" dataset generated with numpy.
import numpy as np

rng = np.random.default_rng(42)

def make_blobs(n_per_class=200):
    # 3 Gaussian blobs in 2D
    centers = np.array([[0,0],[3,3],[-3,3]], dtype=float)
    X_list, y_list = [], []
    for cls, c in enumerate(centers):
        Xc = rng.normal(loc=c, scale=0.8, size=(n_per_class, 2))
        yc = np.full((n_per_class,), cls, dtype=int)
        X_list.append(Xc); y_list.append(yc)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def one_hot(y, num_classes):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1.0
    return oh

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # stability
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)

def cross_entropy(probs, y_true_oh):
    # mean cross-entropy
    eps = 1e-12
    return -np.mean(np.sum(y_true_oh * np.log(probs + eps), axis=1))

# ----- create data -----
X, y = make_blobs(n_per_class=300)
num_classes = 3
y_oh = one_hot(y, num_classes)
n_samples, n_features = X.shape

# ----- init parameters -----
hidden = 16
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.1, size=(n_features, hidden))
b1 = np.zeros((1, hidden))
W2 = rng.normal(0, 0.1, size=(hidden, num_classes))
b2 = np.zeros((1, num_classes))

# ----- training -----
lr = 0.1
epochs = 200
batch_size = 64

for epoch in range(1, epochs+1):
    # mini-batch SGD
    idx = rng.permutation(n_samples)
    X_shuf, y_shuf, y_oh_shuf = X[idx], y[idx], y_oh[idx]

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        xb = X_shuf[start:end]
        yb = y_shuf[start:end]
        yb_oh = y_oh_shuf[start:end]

        # forward
        z1 = xb @ W1 + b1
        a1 = relu(z1)
        logits = a1 @ W2 + b2
        probs = softmax(logits)

        # loss
        loss = cross_entropy(probs, yb_oh)

        # backward
        # dL/dlogits = probs - y
        dlogits = (probs - yb_oh) / xb.shape[0]
        dW2 = a1.T @ dlogits
        db2 = dlogits.sum(axis=0, keepdims=True)

        da1 = dlogits @ W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = xb.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # monitor
    if epoch % 20 == 0 or epoch == 1:
        # full-batch eval
        z1 = X @ W1 + b1
        a1 = relu(z1)
        logits = a1 @ W2 + b2
        probs = softmax(logits)
        preds = probs.argmax(axis=1)
        acc = (preds == y).mean()
        ce = cross_entropy(probs, y_oh)
        print(f"Epoch {epoch:3d} | loss={ce:.4f} | acc={acc*100:.2f}%")

# final
z1 = X @ W1 + b1
a1 = relu(z1)
logits = a1 @ W2 + b2
probs = softmax(logits)
preds = probs.argmax(axis=1)
acc = (preds == y).mean()
print(f"Final accuracy: {acc*100:.2f}%")
