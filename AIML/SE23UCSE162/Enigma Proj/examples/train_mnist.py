# examples/train_mnist_auto.py
# MNIST end-to-end using stdlib + numpy + pandas.
# Robust download with mirrors; falls back to mnist.npz -> CSV conversion.

import os, gzip, urllib.request, pathlib, sys, random
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mynn import Sequential, Dense, ReLU, CrossEntropyLoss, SGD, accuracy

# Config
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "mnist_test.csv")

IDX_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",      # PyTorch mirror
    "https://storage.googleapis.com/cvdf-datasets/mnist/", # Google CVDF mirror
]
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
NPZ_PATH = os.path.join(DATA_DIR, "mnist.npz")

# Utils
def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def try_download(url, dest):
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"[download failed] {e}")
        return False

def load_idx_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    magic, num, rows, cols = np.frombuffer(data[:16], dtype=">i4")
    assert magic == 2051, f"Bad magic for images: {magic}"
    imgs = np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows*cols)
    return imgs

def load_idx_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    magic, num = np.frombuffer(data[:8], dtype=">i4")
    assert magic == 2049, f"Bad magic for labels: {magic}"
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def build_csv_from_idx(limit_train=None, limit_test=None):
    # Try each mirror for each file
    paths = {}
    for key, fname in IDX_FILES.items():
        dest = os.path.join(DATA_DIR, fname)
        if not os.path.exists(dest):
            ok = False
            for mirror in IDX_MIRRORS:
                if try_download(mirror + fname, dest):
                    ok = True
                    break
            if not ok:
                return False  
        paths[key] = dest

    # Parse IDX
    Xtr = load_idx_images(paths["train_images"])
    ytr = load_idx_labels(paths["train_labels"])
    Xte = load_idx_images(paths["test_images"])
    yte = load_idx_labels(paths["test_labels"])

    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]

    # Write CSVs: label in col 0, pixels in 1..784
    pd.DataFrame(np.column_stack([ytr, Xtr])).to_csv(TRAIN_CSV, header=False, index=False)
    pd.DataFrame(np.column_stack([yte, Xte])).to_csv(TEST_CSV, header=False, index=False)
    print(f"Wrote {TRAIN_CSV} and {TEST_CSV} from IDX")
    return True

def build_csv_from_npz(limit_train=None, limit_test=None):
    if not os.path.exists(NPZ_PATH):
        if not try_download(NPZ_URL, NPZ_PATH):
            return False
    with np.load(NPZ_PATH) as data:
        Xtr, ytr = data["x_train"], data["y_train"]
        Xte, yte = data["x_test"],  data["y_test"]
    Xtr = Xtr.reshape(-1, 28*28)
    Xte = Xte.reshape(-1, 28*28)
    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    pd.DataFrame(np.column_stack([ytr, Xtr])).to_csv(TRAIN_CSV, header=False, index=False)
    pd.DataFrame(np.column_stack([yte, Xte])).to_csv(TEST_CSV, header=False, index=False)
    print(f"Wrote {TRAIN_CSV} and {TEST_CSV} from NPZ")
    return True

def maybe_prepare_csvs(limit_train=None, limit_test=None):
    ensure_dir(DATA_DIR)
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        return TRAIN_CSV, TEST_CSV

    # Trying IDX mirrors first, then NPZ
    if build_csv_from_idx(limit_train, limit_test):
        return TRAIN_CSV, TEST_CSV
    print("IDX mirrors failed, falling back to mnist.npz")
    if build_csv_from_npz(limit_train, limit_test):
        return TRAIN_CSV, TEST_CSV
    raise RuntimeError("Failed to download MNIST from all sources.")

def one_hot(y, C):
    oh = np.zeros((y.size, C), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None

# Main flow
if __name__ == "__main__":
    train_csv, test_csv = maybe_prepare_csvs(limit_train=20000, limit_test=5000)

    # Load CSVs with pandas (rule-compliant)
    tr = pd.read_csv(train_csv, header=None).values
    te = pd.read_csv(test_csv, header=None).values

    ytr = tr[:, 0].astype(int)
    Xtr = tr[:, 1:].astype(np.float32) / 255.0
    yte = te[:, 0].astype(int)
    Xte = te[:, 1:].astype(np.float32) / 255.0

    C = len(np.unique(ytr))
    ytr_oh = one_hot(ytr, C)

    # Build model (NumPy-only library)
    model = Sequential()
    model.add(Dense(in_features=784, out_features=128))
    model.add(ReLU())
    model.add(Dense(in_features=128, out_features=C))

    loss = CrossEntropyLoss()
    opt = SGD(lr=0.2, weight_decay=0.0)

    model.compile(loss=loss, optimizer=opt)

    # Train
    print("Training on MNIST (subset) ...")
    model.fit(
        Xtr, ytr_oh,
        epochs=10,           
        batch_size=128,
        verbose=1,
        X_val=Xte, y_val=yte, metric_fn=accuracy
    )

    # Evaluate
    test_acc = model.evaluate(Xte, yte, accuracy)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    # Save predictions and optional plots
    logits = model.predict(Xte)
    preds = logits.argmax(axis=1)
    np.savetxt(os.path.join(DATA_DIR, "mnist_test_preds.csv"), preds, fmt="%d", delimiter=",")
    print(f"Saved predictions to {os.path.join(DATA_DIR, 'mnist_test_preds.csv')}")

    # Optional visuals (only when matplotlib is installed)
    plt = try_import_matplotlib()
    if plt is not None:
        idxs = random.sample(range(Xte.shape[0]), 10)
        fig = plt.figure(figsize=(10, 2))
        for i, idx in enumerate(idxs, 1):
            ax = fig.add_subplot(2, 5, i)
            ax.imshow(Xte[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"pred {preds[idx]}")
            ax.axis("off")
        fig.tight_layout()
        grid_path = os.path.join(DATA_DIR, "mnist_preds.png")
        fig.savefig(grid_path, dpi=150)
        print(f"Saved prediction grid to {grid_path}")
    else:
        print("matplotlib not installed, skipping plots.")
