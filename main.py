from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

class TransactionData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: TransactionData):
    arr = np.array(data.features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)
    prob = model.predict_proba(arr_scaled)[0][1]
    return {
        "prediction": int(prediction[0]),
        "label": "FRAUD" if prediction[0] == 1 else "NOT FRAUD",
        "fraud_probability": round(float(prob), 4)
    }
