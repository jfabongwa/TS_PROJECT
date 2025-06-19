import pandas as pd
from app.config import FEATURES, TARGET

def load_data(train_path: str, test_path: str):
    """Load training and test CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def split_features_target(df: pd.DataFrame):
    """Split features and target from dataframe."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

def get_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features only (used for prediction)."""
    return df[FEATURES]

def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare a single-row DataFrame from user input."""
    return pd.DataFrame([data])[FEATURES]

