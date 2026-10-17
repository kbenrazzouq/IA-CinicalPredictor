"""core/dataset.py

Classe Dataset : récupère les données patients depuis le CSV,
sépare les variables explicatives (X) de la cible (y)
et fournit un découpage train/test.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Racine du projet = dossier parent de core/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")

FEATURES = ["age", "hypertension", "heart_disease", "bmi",
            "HbA1c_level", "blood_glucose_level"]
TARGET = "diabetes"


class Dataset:
    def __init__(self, path=DEFAULT_PATH, features=None, target=TARGET):
        self.path = path
        self.features = features or FEATURES
        self.target = target
        self.data = None

    def load(self):
        """Lit le CSV, vérifie les colonnes, supprime les doublons et les lignes incomplètes."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Fichier introuvable : {self.path}")

        df = pd.read_csv(self.path)

        manquantes = [c for c in self.features + [self.target] if c not in df.columns]
        if manquantes:
            raise ValueError(f"Colonnes absentes du CSV : {manquantes}")

        self.data = (df[self.features + [self.target]]
                     .drop_duplicates()
                     .dropna()
                     .reset_index(drop=True))
        return self

    def get_xy(self):
        """Retourne X et y (tableaux numpy de nombres décimaux)."""
        if self.data is None:
            self.load()
        X = self.data[self.features].to_numpy(dtype=float)
        y = self.data[self.target].to_numpy(dtype=float)
        return X, y

    def split(self, test_size=0.2, seed=42):
        """Retourne (X_train, X_test, y_train, y_test), classes bien réparties."""
        X, y = self.get_xy()
        return train_test_split(X, y, test_size=test_size,
                                random_state=seed, stratify=y)

    def __len__(self):
        return 0 if self.data is None else len(self.data)


if __name__ == "__main__":
    ds = Dataset().load()
    X_train, X_test, y_train, y_test = ds.split()
    print(f"{len(ds)} patients | train={len(X_train)} | test={len(X_test)}")
    print(f"Taux de diabète : {y_train.mean():.2%}")
