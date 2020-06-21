import numpy as np

from utils.losses import squared_error


def gradient_descent(
        X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1,
        iterations: int = 100):
    # Initialising theta
    theta = np.random.randn(X.shape[1], 1)

    m = len(y)
    theta_h = np.zeros((iterations, theta.shape[0]))
    cost_h = np.zeros(iterations)

    for i in range(iterations):
        pred = X.dot(theta)
        theta -= learning_rate / m * X.T.dot(pred - y)
        theta_h[i, :] = theta.T
        cost_h[i] = squared_error(pred, y)

        print("Iteration: %s" % i)
        print("Cost: %s" % np.round(cost_h[i], 4))

    return theta_h, cost_h
