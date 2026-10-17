from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle."""

    @abstractmethod
    def predict(self, X):
        """Retourne la probabilité (entre 0 et 1) d'être infecté, pour chaque ligne."""

    def predict_class(self, X, threshold=0.5):
        """Retourne la classe 0/1 en appliquant le seuil."""
        return (self.predict(X) >= threshold).astype(int)
