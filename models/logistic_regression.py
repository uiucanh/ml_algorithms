import numpy as np

from models.base import BaseModel
from utils.optimizers import gradient_descent
from utils.losses import binary_cross_entropy
from utils.activation import sigmoid


class LogisticRegression(BaseModel):
    def __init__(self, n_iter=100):
        super().__init__(name='Logistic Regression', n_iter=n_iter)

    def preprocess(self, X: np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self.preprocess(X)
        print("Fitting %s" % self.name)

        theta_h, cost_h = gradient_descent(
            X, y, cost_function=binary_cross_entropy, iterations=self.n_iter,
            activation_func=sigmoid
        )
        self.cost_h = cost_h
        self.beta_hat = theta_h[np.argmin(cost_h)]

    def predict_proba(self, X: np.ndarray):
        X = self.preprocess(X)

        y_preds = X.dot(self.beta_hat)
        probs = sigmoid(y_preds)
        return probs

    def predict(self, X: np.ndarray, thres: float = 0.5):
        probs = self.predict_proba(X)
        probs[probs > thres] = 1
        probs[probs <= thres] = 0
        return probs.reshape(-1, 1)

    def score(self, y_test: np.ndarray, y_pred: np.ndarray,
              metric: str = 'F1'):
        result = self._score(y_test, y_pred, metric=metric)
        print("%s score: %s" % (metric, np.round(result, 2)))
        return result
