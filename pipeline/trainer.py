
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import Preprocessing
from core.dataset import Dataset


class Trainer:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path
        self.scaler = StandardScaler()

    def train(self):
        preprocessing = Preprocessing(self.data_path)
        X, y = preprocessing.load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        # Normaliser les caractéristiques
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Créer le Dataset d'entraînement
        train_dataset = Dataset(X_train, y_train)
        X_train, y_train = train_dataset.get_data()

        # Entraîner le modèle
        self.model.fit(X_train, y_train)

        return self.model, X_test, y_test
