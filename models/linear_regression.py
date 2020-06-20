import numpy as np

from models.base import BaseModel


class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__(name='Linear Regression')
        self.beta_hat = 0
        self.bias = 0
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, method='ols'):
        print("Fitting %s" % self.name)
        if method == 'ols':
            self.beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            self.bias = np.mean(y) - np.dot(self.beta_hat, np.mean(X, axis=0))
        elif method == 'gradient_descent':
            pass
        else:
            print("Only 'ols' and 'gradient_descent' are available")
        self.fitted = True

    def predict(self, X: np.ndarray):
        if not self.fitted:
            print("Model hasn't been fitted")
            return

        y_pred = self.bias + np.dot(X, self.beta_hat)
        return y_pred

    def score(self, y_test: np.ndarray, y_pred: np.ndarray,
              metric: str = 'R2'):
        result = self._score(y_test, y_pred, metric=metric)
        print("%s score: %s" % (metric, np.round(result, 2)))
        return result
