"""core/optimizer.py

Optimiseurs : mettent à jour les paramètres d'un modèle à partir des gradients.
"""
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, params, grads):
        """Modifie le dictionnaire `params` à partir du dictionnaire `grads`."""


class GradientDescent(Optimizer):
    """Descente de gradient : param = param - learning_rate * gradient."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for nom in params:
            params[nom] = params[nom] - self.learning_rate * grads[nom]


class Momentum(Optimizer):
    """Descente de gradient avec élan (converge souvent plus vite)."""

    def __init__(self, learning_rate=0.1, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def update(self, params, grads):
        for nom in params:
            v = self.beta * self.velocity.get(nom, 0.0) + grads[nom]
            self.velocity[nom] = v
            params[nom] = params[nom] - self.learning_rate * v
