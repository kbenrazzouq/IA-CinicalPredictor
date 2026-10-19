from utils.metrics import Metrics


class Evaluator:

    def __init__(self):
        self.metrics = Metrics()

    def evaluate(self, model, X_test, y_test):
        probabilities = model.predict(X_test)
        predictions = (probabilities >= 0.5).astype(int)

        scores = self.metrics.calculate(y_test, predictions)

        for name, value in scores.items():
            print(f"{name} : {value:.3f}")

        return scores