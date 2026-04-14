from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"status": "API working 🚀"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(list(data.values())).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]

    return {"fraud_probability": float(prob)}