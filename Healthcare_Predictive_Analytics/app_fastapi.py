# """
# FastAPI version for Healthcare Predictive Analytics Deployment
# ---------------------------------------------------------------
# Run:
#     uvicorn app_fastapi:app --reload
# Then open: http://127.0.0.1:8000/docs
# """

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Healthcare Predictive Analytics API")

model_path = os.path.join("artifacts", "model_randomforest.joblib")
scaler_path = os.path.join("artifacts", "scaler.joblib")

model = joblib.load(model_path) if os.path.exists(model_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


class PatientData(BaseModel):
    features: list


@app.get("/")
def root():
    return {"message": "Healthcare Predictive Analytics API - Ready"}


@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array([data.features])
        if scaler:
            features = scaler.transform(features)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None

        return {"prediction": int(prediction), "probability": proba}
    except Exception as e:
        return {"error": str(e)}
