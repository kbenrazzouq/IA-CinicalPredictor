import numpy as np
class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)   # évite la division par 0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Appelez fit() avant transform().")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        return np.asarray(X_scaled, dtype=float) * self.std_ + self.mean_


def parse_patient(patient_data, features):
    if isinstance(patient_data, dict):
        valeurs = [patient_data.get(f) for f in features]
    else:
        valeurs = list(patient_data)

    if len(valeurs) != len(features):
        raise ValueError(f"{len(features)} valeurs attendues : {features}")
    try:
        ligne = [float(str(v).strip().replace(",", ".")) for v in valeurs]
    except ValueError:
        raise ValueError(f"Valeur manquante ou non numérique dans : {valeurs}")
    return np.array(ligne).reshape(1, -1)
