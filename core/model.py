
from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def fit(self, X, y):
        """Entraîner le modèle."""
        pass

    @abstractmethod
    def predict(self, X):
        """Prédire les probabilités."""
        pass
