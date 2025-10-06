# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_data

app = FastAPI(title="Breast Cancer Classifier API", version="1.0.0")

class CancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_smoothness: float
    mean_compactness: float

@app.get("/")
def root():
    return {"message": "Breast Cancer Classifier API is running!"}

@app.post("/predict")
async def predict_cancer(data: CancerData):
    features = [[
        data.mean_radius,
        data.mean_texture,
        data.mean_smoothness,
        data.mean_compactness
    ]]
    prediction = predict_data(features)
    label = "Benign" if prediction[0] == 1 else "Malignant"
    return {"predicted_class": int(prediction[0]), "label": label}