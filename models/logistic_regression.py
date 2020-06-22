import numpy as np

from models.base import BaseModel
from utils.optimizers import gradient_descent
from utils.losses import binary_cross_entropy
from utils.activation import sigmoid


class LogisticRegression(BaseModel):
    def __init__(self, n_iter=100):
        super().__init__(name='Logistic Regression', n_iter=n_iter)

    def preprocess(self, X: np.ndarray, y: np.ndarray):
        return

    def fit(self, X: np.ndarray, y: np.ndarray):
        theta_h, cost_h = gradient_descent(
            X, y, cost_function=binary_cross_entropy, iterations=self.n_iter,
            activation_func=sigmoid
        )
        self.cost_h = cost_h
        self.beta_hat = theta_h[np.argmin(cost_h)]

    def predict(self, X: np.ndarray, thres: float = 0.5):
        y_preds = X.dot(self.beta_hat)
        y_preds = sigmoid(y_preds)
        y_preds[y_preds > thres] = 1
        y_preds[y_preds <= thres] = 0
        return y_preds.reshape(-1, 1)

    def score(self, y_test: np.ndarray, y_pred: np.ndarray,
              metric: str = 'accuracy'):
        result = self._score(y_test, y_pred, metric=metric)
        print("%s score: %s" % (metric, np.round(result, 2)))
        return result
