import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

# Set experiment name (you can change this)
#mlflow.set_tracking_uri("http://localhost:5000")  # Or omit to use default local URI
mlflow.set_experiment("Retail Demand Forecast")

# Load pre-split datasets
train_df = pd.read_csv("data/train_modelling.csv")
test_df = pd.read_csv("data/test_modelling.csv")

# Define features and target
FEATURES = ['weekday', 'weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'onpromotion']
TARGET = 'unit_sales'

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

with mlflow.start_run(run_name="xgboost_final_v1"):
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2_score": r2})

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = "model/xgboost_best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Log model artifact
    mlflow.log_artifact(model_path)

    print(f"\nRun complete! MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
