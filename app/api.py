from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer
from pipeline.predictor import ClinicalPredictor


class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


class ClinicalAPI:

    def __init__(self):
        self.app = FastAPI(title="Diagnostic du diabète")

        trainer = Trainer(
            LogisticRegression(),
            "data/diabetes_prediction_dataset.csv"
        )

        model, _, _ = trainer.train()
        self.predictor = ClinicalPredictor(model, trainer.scaler)

        self.app.post("/diagnose")(self.diagnose)

    def diagnose(self, patient: PatientData):
        data = pd.DataFrame([patient.model_dump()])
        result = self.predictor.diagnose(data)

        return {"diagnostic": result}


clinical_api = ClinicalAPI()
app = clinical_api.app