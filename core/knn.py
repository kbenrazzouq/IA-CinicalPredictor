import numpy as np

from core.model import Model


class KNN(Model):
    """Model des k plus proches voisins (classification binaire 0/1)."""

    def __init__(self, k=5):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k doit être un entier strictement positif.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Entraîner le modèle (KNN mémorise simplement les données)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """Prédire les probabilités de la classe 1."""
        if self.X_train is None:
            raise RuntimeError("Le modèle doit être entraîné avec fit() avant predict().")

        X = np.array(X)
        probabilities = []

        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest = np.argsort(distances)[:self.k]
            probabilities.append(np.mean(self.y_train[nearest]))

        return np.array(probabilities)