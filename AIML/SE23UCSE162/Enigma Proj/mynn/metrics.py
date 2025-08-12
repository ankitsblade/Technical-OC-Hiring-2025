import numpy as np

def accuracy(logits, y_true):
    preds = logits.argmax(axis=1)
    return (preds == y_true).mean()
