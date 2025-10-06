# src/data.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

def get_data():
    """
    Loads the breast cancer dataset and returns
    it as (X, y, feature_names, target_names).
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.feature_names, data.target_names

if __name__ == "__main__":
    X, y, features, targets = get_data()
    print("✅ Data loaded successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Target names: {list(targets)}")