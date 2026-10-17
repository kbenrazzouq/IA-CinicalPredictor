"""core/logistic_regression.py"""
import numpy as np

from core.model import Model
from core.optimizer import GradientDescent


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(Model):
    def __init__(self, optimizer=None, epochs=500, l2=0.0):
        self.optimizer = optimizer or GradientDescent(learning_rate=0.1)
        self.epochs = epochs
        self.l2 = l2                      # régularisation (0 = aucune)
        self.params = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.params = {"w": np.zeros(d), "b": np.array(0.0)}
        self.loss_history = []

        for _ in range(self.epochs):
            p = self.predict(X)
            erreur = p - y
            grads = {"w": X.T @ erreur / n + self.l2 * self.params["w"],
                     "b": np.array(erreur.mean())}
            self.optimizer.update(self.params, grads)
            self.loss_history.append(self._loss(y, p))
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Modèle non entraîné : appelez fit() d'abord.")
        return sigmoid(np.asarray(X, dtype=float) @ self.params["w"] + self.params["b"])

    @staticmethod
    def _loss(y, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
