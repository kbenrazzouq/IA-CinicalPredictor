
class ClinicalPredictor:

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def diagnose(self, patient_data):
        # Appliquer la même normalisation que pendant l'entraînement
        patient_scaled = self.scaler.transform(patient_data)

        # Obtenir la probabilité de diabète
        probability = self.model.predict(patient_scaled)[0]

        # Déterminer la classe prédite
        if probability >= 0.5:
            return "Diabétique"
        else:
            return "Non diabétique"
