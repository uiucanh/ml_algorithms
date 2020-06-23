import matplotlib.pyplot as plt
import numpy as np

from models.logistic_regression import LogisticRegression


def plot_regression(X, y_test, y_pred):
    plt.scatter(X, y_test, c='blue')
    plt.plot(X, y_pred, c='red')
    plt.show()


def plot_regression_residual(y_test, y_pred, bins=10):
    error = y_test - y_pred
    plt.hist(error, bins=bins)
    plt.show()


def plot_iteration_vs_cost(n_iter: int, cost_h: np.ndarray):
    plt.scatter(np.arange(1, n_iter + 1, 1), cost_h.reshape(1, -1).tolist())
    plt.show()


def plot_points_and_cluster(X: np.ndarray, centers: np.ndarray):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1])
    plt.show()


def plot_logistic_regression_decision_boundary(
        X: np.ndarray, y: np.ndarray, model: LogisticRegression):
    neg_mask = y.reshape(1, -1) == 0
    pos_mask = y.reshape(1, -1) == 1
    neg_mask = neg_mask[0]
    pos_mask = pos_mask[0]

    plt.scatter(X[neg_mask][:, 0], X[neg_mask][:, 1])
    plt.scatter(X[pos_mask][:, 0], X[pos_mask][:, 1], c='red')
    x_values = [np.min(X[:, 0] - 1), np.max(X[:, 1] + 1)]
    y_values = -(model.beta_hat[0] + np.dot(model.beta_hat[1], x_values)) / \
               model.beta_hat[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.show()
