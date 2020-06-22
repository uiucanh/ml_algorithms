import numpy as np


def squared_error(y_preds: np.ndarray, y: np.ndarray):
    m = len(y)

    loss = 1 / m * np.square(y_preds - y).sum()
    return loss


def binary_cross_entropy(y_preds: np.ndarray, y: np.ndarray):
    m = len(y)

    loss = 1 / m * (-y.T.dot(y_preds) - (1 - y).T.dot(np.log(1 - y_preds)))
    return loss
