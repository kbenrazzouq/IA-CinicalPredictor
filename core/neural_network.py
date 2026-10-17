"""core/neural_network.py

Réseau de neurones : 1 couche cachée (tanh) + sortie sigmoïde.
"""
import numpy as np

from core.logistic_regression import sigmoid
from core.model import Model
from core.optimizer import GradientDescent


class NeuralNetwork(Model):
    def __init__(self, hidden_units=8, optimizer=None, epochs=500, seed=42):
        self.hidden_units = hidden_units
        self.optimizer = optimizer or GradientDescent(learning_rate=0.1)
        self.epochs = epochs
        self.seed = seed
        self.params = None
        self.loss_history = []

    def _forward(self, X):
        a1 = np.tanh(X @ self.params["W1"] + self.params["b1"])
        p = sigmoid(a1 @ self.params["W2"] + self.params["b2"])
        return a1, p

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        h = self.hidden_units
        self.params = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, h)),
                       "b1": np.zeros(h),
                       "W2": rng.normal(0, 1 / np.sqrt(h), h),
                       "b2": np.array(0.0)}
        self.loss_history = []

        for _ in range(self.epochs):
            a1, p = self._forward(X)
            # Rétropropagation
            dz2 = (p - y) / n
            dz1 = (dz2[:, None] * self.params["W2"][None, :]) * (1 - a1 ** 2)
            grads = {"W2": a1.T @ dz2, "b2": np.array(dz2.sum()),
                     "W1": X.T @ dz1, "b1": dz1.sum(axis=0)}
            self.optimizer.update(self.params, grads)

            pc = np.clip(p, 1e-12, 1 - 1e-12)
            self.loss_history.append(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        return self

    def predict(self, X):
        if self.params is None:
            raise RuntimeError("Modèle non entraîné : appelez fit() d'abord.")
        return self._forward(np.asarray(X, dtype=float))[1]
