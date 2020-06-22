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
    tp = np.logical_and(y_pred == y_test, y_pred == positive).sum()
    fp = np.logical_and(y_pred != y_test, y_pred == positive).sum()
    return tp / (tp + fp)


def recall(y_test, y_pred, positive=1):
    tp = np.logical_and(y_pred == y_test, y_pred == positive).sum()
    fn = np.logical_and(y_pred != y_test, y_pred != positive).sum()
    return tp / (tp + fn)


def f1(y_test, y_pred, positive=1):
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    return 2 * (p * r) / (p + r)
