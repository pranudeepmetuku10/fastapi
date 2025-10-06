# src/predict.py
import joblib
import numpy as np

# Load trained model
model = joblib.load("../model/cancer_model.pkl")

def predict_data(features):
    """
    Takes input features [[...]] and returns model predictions.
    """
    features = np.array(features)
    prediction = model.predict(features)
    return prediction