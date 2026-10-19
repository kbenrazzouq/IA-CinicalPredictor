from sklearn.neural_network import MLPClassifier
from core.model import Model


class NeuralNetwork(Model):

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=1000,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]