import matplotlib.pyplot as plt
import numpy as np


def plot_regression(X, y_test, y_pred):
    plt.scatter(X, y_test, c='blue')
    plt.plot(X, y_pred, c='red')
    plt.show()


def plot_regression_residual(y_test, y_pred, bins=10):
    error = y_test - y_pred
    plt.hist(error, bins=bins)
    plt.show()
