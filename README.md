#  Lab 2 – MLOps (IE-7374)

## Overview
This lab focused on exposing a trained machine learning model as a REST API using **FastAPI** and **Uvicorn**.  
Instead of using the traditional Iris dataset, this implementation uses the **Breast Cancer dataset** from `scikit-learn` to build a binary classifier that predicts whether a tumor is *benign* or *malignant*.

The objective was to go beyond model training by learning how to **serve models in real time** — a key step in the modern **MLOps workflow**.

---

##  Project Structure

```
lab_2/
│── model/
│   └── cancer_model.pkl               # trained Decision Tree model
│── src/
│   ├── data.py                        # loads Breast Cancer dataset
│   ├── train.py                       # trains and saves model
│   ├── predict.py                     # loads model and performs predictions
│   └── main.py                        # FastAPI app exposing prediction endpoint
│── requirements.txt                   # dependencies
│── README.md                          # setup and usage guide
│── .gitignore                         # excluded files and folders

```

---

## Workflow

###  Data Loading (`data.py`)
- Implemented a reusable function `get_data()` that loads the Breast Cancer dataset as Pandas DataFrames and returns `(X, y)` along with feature names.  
- Makes it easy to switch datasets later without modifying training or inference code.

### Model Training (`train.py`)
- Trained a **DecisionTreeClassifier** using 4 key features:  
  `["mean radius", "mean texture", "mean smoothness", "mean compactness"]`
- Saved the trained model as `cancer_model.pkl` using `joblib`.  
- Designed so training can be rerun independently before deployment.

###  Prediction Logic (`predict.py`)
- Loads the model and performs predictions via a function `predict_data()`.  
- Keeps inference logic separate from web serving code for cleaner maintainability.

### API Development (`main.py`)
- Built a **FastAPI** application with two endpoints:
  - `GET /` → health check  
  - `POST /predict` → accepts JSON input and returns prediction
- Defined a **Pydantic model** for input validation (`CancerData`).
- Added human-readable outputs ("Benign" / "Malignant") for clarity.

###  Model Serving
Run the API with:
bash
uvicorn main:app --reload

## What I Changed / Improved
- Replaced the default Iris model with a **Breast Cancer classifier** while keeping the FastAPI architecture identical.  
- Simplified API inputs to **4 representative numeric features** for cleaner and faster testing.  
- Modularized dataset handling with `data.py` to improve reusability and separation of concerns.  
- Added **clear logging and success messages** during model training for better traceability.  
- Created a refined `.gitignore` and `requirements.txt` to ensure a clean and professional GitHub setup.  


## Results

###  Local Testing
- Model trained successfully and saved as:
- ../model/cancer_model.pkl
-  #### Example input:

*{
  "mean_radius": 14.5,
  "mean_texture": 19.0,
  "mean_smoothness": 0.10,
  "mean_compactness": 0.14
}*

- #### Example output:
*{
  "predicted_class": 1,
  "label": "Benign"
}*

### Swagger UI
	•	FastAPI auto-generated interactive API documentation available at:
  http://127.0.0.1:8000/docs
  	**#### Successfully tested the /predict endpoint with multiple inputs**

  ### GitHub Integration
	•	Repository created and pushed: fastapi
	•	.gitignore excludes environment folders, model files, and cache.
	•	README.md structured to reflect professional MLOps documentation standards.



