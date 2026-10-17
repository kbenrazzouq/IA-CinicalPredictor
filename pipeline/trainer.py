import os
import joblib
from utils.preprocessing import StandardScaler


class Trainer:
    def __init__(self, model, dataset, scaler=None):
        self.model = model
        self.dataset = dataset
        self.scaler = scaler or StandardScaler()
        self.X_test = None
        self.y_test = None

    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.split()
        X_train = self.scaler.fit_transform(X_train)      # fit sur le train seulement
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test
        self.model.fit(X_train, y_train)
        return self.model

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model,
                     "scaler": self.scaler,
                     "features": self.dataset.features}, path)
