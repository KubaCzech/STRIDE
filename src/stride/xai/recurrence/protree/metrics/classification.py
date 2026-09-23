import numpy as np


def balanced_accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    acc = 0.0
    for label in np.unique(y):
        mask = y == label
        acc_partial = (y[mask] == y_hat[mask]).mean()
        acc += (acc_partial * mask.sum()) / len(y)
    return acc.item()
