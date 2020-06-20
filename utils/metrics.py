import numpy as np

available_metrics = ['R2']


def r2(y_test, y_pred):
    # Reshaping to match shapes
    y_pred = y_pred.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    y_mean = np.mean(y_test)
    ss_tot = np.square(y_test - y_mean).sum()
    ss_res = np.square(y_test - y_pred).sum()
    result = 1 - ss_res / ss_tot
    return result
