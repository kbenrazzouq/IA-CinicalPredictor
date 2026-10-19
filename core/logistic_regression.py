from sklearn.linear_model import LogisticRegression as SklearnLR
from core.model import Model


class LogisticRegression(Model):

    def __init__(self):
        self.model = SklearnLR(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]