import sys
import os

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")

from models.linear_regression import LinearRegression
from utils.data_utils import generate_linear_data, split_dataset
from utils.plot_utils import plot_regression_residual


def main():
    X, y = generate_linear_data(n_samples=1000, n_features=10)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print("Training size: %s, Test size %s" % (len(X_train), len(X_test)))
    print("-" * 20)

    # Fit and predict
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("-" * 20)

    # Scoring
    model.score(y_test, y_pred)
    print("-" * 20)

    # Plotting
    plot_regression_residual(y_test, y_pred, bins=int(len(X_train) / 20))


if __name__ == '__main__':
    main()
