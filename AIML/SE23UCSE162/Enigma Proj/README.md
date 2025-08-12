# NumPy Neural Network

## Overview
This project implements a simple feedforward neural network using only **NumPy** and **Pandas**, without relying on deep learning frameworks like TensorFlow or PyTorch.

It was built in three main stages:
1. **Basic Neural Network Script** – A single Python script that can train a small neural network for a simple classification task.
2. **Custom Library** – Refactored into an object-oriented, PyTorch-like mini-framework with `Sequential`, `Dense`, `ReLU`, `Softmax`, `CrossEntropyLoss`, and `SGD`.
3. **Real-World Demonstration** – Trained the custom library on the **MNIST** handwritten digit classification dataset.

---

## Features
- Fully implemented forward and backward propagation using NumPy.
- Modular structure for adding new layers and activation functions.
- Training loop with batch updates, validation, and accuracy tracking.
- Works on **any CSV-formatted dataset** with numeric features and labels.

---

## File Structure
```
Enigma Proj/
│
├── mynn/                  # Core library code
│   ├── __init__.py
│   ├── layers.py          # Dense, ReLU, Softmax
│   ├── losses.py          # CrossEntropyLoss
│   ├── model.py           # Sequential model handling
│   ├── optim.py           # SGD optimizer
│
├── examples/
│   ├── train_blobs.py     # Simple 2D classification demo
│   ├── train_mnist.py # MNIST training script
│
└── data/                  # MNIST CSVs and prediction outputs
```

---

## How to Run

### Train on a Toy Dataset
```bash
python -m examples.train_blobs
```

### Train on MNIST (Auto Download + CSV Conversion)
```bash
python -m examples.train_mnist_auto
```

The script automatically downloads MNIST, converts it to CSV, and trains the model.

---

## Example Output (MNIST)
```
Epoch 001 | loss=0.2255 | val_acc=87.56%
Epoch 002 | loss=0.4715 | val_acc=87.96%
...
Test accuracy: 93.98%
```

---

## Next Steps / Future Work
- Add more activation functions (Tanh, LeakyReLU, etc.)
- Implement other optimizers (Adam, RMSprop)
- Add convolution layers for CNN support
- Save and load trained models

---

## Credits
Built with **NumPy** and **Pandas**.
MNIST dataset courtesy of Yann LeCun.
Official NumPy Documentation.
Youtube.