
import pandas as pd


class ClinicalInterface:
    def __init__(self, predictor):
        self.predictor = predictor

    def run(self):
        print("\n=== Diagnostic du diabète ===")

        age = float(input("Âge : "))
        hypertension = int(input("Hypertension (1=oui, 0=non) : "))
        heart_disease = int(input("Maladie cardiaque (1=oui, 0=non) : "))
        bmi = float(input("IMC (BMI) : "))
        hba1c = float(input("Taux d'HbA1c : "))
        glucose = float(input("Glycémie (blood glucose level) : "))

        patient_data = pd.DataFrame([{
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        result = self.predictor.diagnose(patient_data)

        print("\nRésultat du modèle :", result)
        print("Attention : résultat pédagogique, non médicalement validé.")
