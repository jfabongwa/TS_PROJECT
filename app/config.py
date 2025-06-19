MODEL_PATH = "model/xgboost_best_model.pkl"
TRAIN_PATH = "data/train_modelling.csv"
TEST_PATH = "data/test_modelling.csv"

FEATURES = ['weekday', 'weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'onpromotion']
TARGET = 'unit_sales'
