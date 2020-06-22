import numpy as np

available_metrics = ['R2', 'F1', 'accuracy']


def r2(y_test, y_pred):
    y_mean = np.mean(y_test)
    ss_tot = np.square(y_test - y_mean).sum()
    ss_res = np.square(y_test - y_pred).sum()
    result = 1 - ss_res / ss_tot
    return result


def accuracy(y_test, y_pred):
    return np.count_nonzero(y_pred == y_test) / len(y_test)


def precision(y_test, y_pred, positive=1):
    return


def recall(y_test, y_pred, positive=1):
    return


def f1(y_test, y_pred, positive=1):
    return
