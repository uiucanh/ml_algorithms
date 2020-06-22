import argparse
import numpy as np
import sys
import os

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")
np.set_printoptions(precision=3, suppress=True)

from models.logistic_regression import LogisticRegression  # noqa
from utils.data_utils import generate_classification_data, split_dataset  # noqa


def main():
    parser = argparse.ArgumentParser(description='Linear Regression test')
    parser.add_argument('-n', '--n_iter', type=int, default=50,
                        help='number of iterations for grad_descent')
    args = parser.parse_args()
    n_iter = args.n_iter

    X, y, centers = generate_classification_data()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print("Training size: %s, Test size %s" % (len(X_train), len(X_test)))
    print("-" * 20)

    # Fit and predict
    model = LogisticRegression(n_iter=n_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("-" * 20)

    # Scoring
    model.score(y_test, y_pred)
    print("-" * 20)

    # Plotting


if __name__ == '__main__':
    main()
