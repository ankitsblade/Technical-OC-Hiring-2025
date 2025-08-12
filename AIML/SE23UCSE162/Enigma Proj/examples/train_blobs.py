# examples/train_blobs.py
import numpy as np
from mynn import Sequential, Dense, ReLU, Softmax, CrossEntropyLoss, SGD, accuracy

rng = np.random.default_rng(42)

def make_blobs(n_per_class=300):
    centers = np.array([[0,0],[3,3],[-3,3]], dtype=float)
    X_list, y_list = [], []
    for cls, c in enumerate(centers):
        Xc = rng.normal(loc=c, scale=0.8, size=(n_per_class, 2))
        yc = np.full((n_per_class,), cls, dtype=int)
        X_list.append(Xc); y_list.append(yc)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def one_hot(y, C):
    oh = np.zeros((y.size, C)); oh[np.arange(y.size), y] = 1.0
    return oh

# data
X, y = make_blobs(400)
C = len(np.unique(y))
N = X.shape[0]
split = int(0.8*N)
Xtr, ytr = X[:split], y[:split]
Xva, yva = X[split:], y[split:]
ytr_oh = one_hot(ytr, C)

# model
model = Sequential()
model.add(Dense(in_features=2, out_features=32))
model.add(ReLU())
model.add(Dense(in_features=32, out_features=C))  # logits

loss = CrossEntropyLoss()
opt = SGD(lr=0.1, weight_decay=0.0)
model.compile(loss=loss, optimizer=opt)

model.fit(Xtr, ytr_oh, epochs=80, batch_size=64, verbose=1,
          X_val=Xva, y_val=yva, metric_fn=accuracy)

final_acc = model.evaluate(Xva, yva, accuracy)
print(f"Validation accuracy: {final_acc*100:.2f}%")
