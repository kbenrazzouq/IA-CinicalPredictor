
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from app.interface_clinique import ClinicalInterface


class DiagnosticApplication:

    def __init__(self):
        self.model = LogisticRegression()
        self.trainer = Trainer(
            model=self.model,
            data_path="data/diabetes_prediction_dataset.csv"
        )

    def run(self):
        # 1. Entraîner le modèle
        print("\n=== Entraînement du modèle ===")

        model, X_test, y_test = self.trainer.train()

        print("Entraînement terminé.")

        # 2. Évaluer le modèle
        print("\n=== Évaluation du modèle ===")

        evaluator = Evaluator()
        evaluator.evaluate(model, X_test, y_test)

        # 3. Créer le prédicteur avec le scaler entraîné
        predictor = ClinicalPredictor(
            model=model,
            scaler=self.trainer.scaler
        )

        # 4. Lancer l'interface clinique
        interface = ClinicalInterface(predictor)
        interface.run()


if __name__ == "__main__":
    application = DiagnosticApplication()
    application.run()
