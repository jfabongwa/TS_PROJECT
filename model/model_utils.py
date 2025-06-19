import joblib
from app.config import MODEL_PATH
import pickle

import pickle

def load_model(model_path="model/xgboost_best_model.pkl"):
    """Loads the trained model using pickle."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, X):
    return model.predict(X)
