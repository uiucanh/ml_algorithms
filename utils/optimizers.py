import numpy as np

from typing import AnyStr, Callable
from utils.losses import squared_error


def gradient_descent(
        X: np.ndarray, y: np.ndarray, cost_function: Callable,
        learning_rate: float = 0.1, iterations: int = 100,
        activation_func: Callable = None):
    # Initialising theta
    theta = np.random.randn(X.shape[1], 1)

    m = len(y)
    theta_h = np.zeros((iterations, theta.shape[0]))
    cost_h = np.zeros(iterations)

    for i in range(iterations):
        pred = X.dot(theta)
        if activation_func is not None:
            pred = activation_func.__call__(pred)

        theta -= learning_rate / m * X.T.dot(pred - y)
        theta_h[i, :] = theta.T
        cost_h[i] = cost_function.__call__(pred, y)

        print("Iteration: %s" % i)
        print("Cost: %s" % np.round(cost_h[i], 4))

    return theta_h, cost_h
