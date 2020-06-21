import numpy as np


def squared_error(y_preds: np.ndarray, y: np.ndarray):
    m = len(y)

    loss = 1 / m * np.square(y_preds - y).sum()
    return loss