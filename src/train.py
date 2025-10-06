# src/train.py
from data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib, os

def train_model():
    # Load the full dataset
    X, y, feature_names, target_names = get_data()

    # Use only the 4 features your API accepts
    selected_features = ["mean radius", "mean texture", "mean smoothness", "mean compactness"]
    X = X[selected_features]

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, "../model/cancer_model.pkl")
    print("Model trained successfully")
    print(selected_features)

if __name__ == "__main__":
    train_model()