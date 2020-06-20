import numpy as np

from models.base import BaseModel


class LinearRegression(BaseModel):
    def __init__(self, n_iter=100):
        super().__init__(name='Linear Regression', n_iter=n_iter)
        self.beta_hat = 0
        self.fitted = False
        self.cost_h = None

    def preprocess(self, X: np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, method='ols'):
        X = self.preprocess(X)
        print("Fitting %s" % self.name)

        if method == 'ols':
            self.beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            self.beta_hat = self.beta_hat.reshape(X.shape[1])
        elif method == 'grad_descent':
            theta_h, cost_h = gradient_descent(X, y, iterations=self.n_iter)
            self.cost_h = cost_h
            self.beta_hat = theta_h[np.argmin(cost_h)]
        else:
            print("Only 'ols' and 'grad_descent' are available")

        self.fitted = True

    def predict(self, X: np.ndarray):
        X = self.preprocess(X)

        if not self.fitted:
            print("Model hasn't been fitted")
            return

        y_preds = X.dot(self.beta_hat)
        return y_preds.reshape(-1, 1)

    def score(self, y_test: np.ndarray, y_pred: np.ndarray,
              metric: str = 'R2'):
        result = self._score(y_test, y_pred, metric=metric)
        print("%s score: %s" % (metric, np.round(result, 2)))
        return result


def calc_loss(y_preds: np.ndarray, y: np.ndarray):
    m = len(y)

    loss = 1 / (2 * m) * np.square(y_preds - y).sum()
    return loss


def gradient_descent(
        X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1,
        iterations: float = 100):
    # Initialising theta
    theta = np.random.randn(X.shape[1], 1)

    m = len(y)
    theta_h = np.zeros((iterations, theta.shape[0]))
    cost_h = np.zeros(iterations)

    for i in range(iterations):
        pred = X.dot(theta)
        theta -= learning_rate / m * X.T.dot(pred - y)
        theta_h[i, :] = theta.T
        cost_h[i] = calc_loss(pred, y)

        print("Iteration: %s" % i)
        print("Cost: %s" % np.round(cost_h[i], 4))

    return theta_h, cost_h
