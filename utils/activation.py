import numpy as np


def sigmoid(y):
    return 1 / (1 + np.exp(-y))
