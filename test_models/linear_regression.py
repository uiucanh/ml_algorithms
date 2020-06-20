import argparse
import numpy as np
import sys
import os

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")

from models.linear_regression import LinearRegression  # noqa
from utils.data_utils import generate_linear_data, split_dataset  # noqa
from utils.plot_utils import plot_regression_residual  # noqa


def main():
    parser = argparse.ArgumentParser(description='Linear Regression test')
    parser.add_argument('-m', '--method', type=str, default='ols',
                        help='model method: ols or grad_descent')
    args = parser.parse_args()
    method = args.method

    X, y, m, bias = \
        generate_linear_data(n_samples=1000, n_features=10, bias=10)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print("Training size: %s, Test size %s" % (len(X_train), len(X_test)))
    print("-" * 20)

    # Fit and predict
    model = LinearRegression()
    model.fit(X_train, y_train, method)
    y_pred = model.predict(X_test)
    print("-" * 20)

    # Scoring
    model.score(y_test, y_pred)
    print("-" * 20)
    np.set_printoptions(precision=3, suppress=True)
    print("True coefs: ", np.insert(m, 0, bias))
    print("Model coefs:", model.beta_hat)
    print("-" * 20)

    # Plotting
    plot_regression_residual(y_test, y_pred, bins=int(len(X_train) / 20))


if __name__ == '__main__':
    main()
