import numpy as np

from abc import ABC, abstractmethod
from utils.metrics import available_metrics, r2


class BaseModel(ABC):
    def __init__(self, name: str, n_iter: int = 100, seed: int = 0):
        self.name = name
        self.n_iter = n_iter
        self.seed = seed

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        return

    @abstractmethod
    def predict(self, X: np.ndarray):
        return

    @abstractmethod
    def score(self, y_test: np.ndarray, y_pred: np.ndarray, metric: str):
        return

    def _score(self, y_test: np.ndarray, y_pred: np.ndarray, metric: str):
        if metric not in available_metrics:
            print("Not available metric")
            return None

        mapping = {
            'R2': r2
        }

        result = mapping[metric].__call__(y_test, y_pred)
        return result
