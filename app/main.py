import sys
import os

# Add project root to sys.path BEFORE importing from model or data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from model.model_utils import load_model, predict
from data.data_utils import prepare_input
from app.config import FEATURES

st.title("Retail Demand Forecast App")

st.markdown("Enter feature values to predict **unit sales**")

with st.form("prediction_form"):
    inputs = {}
    for feature in FEATURES:
        if feature in ['weekday', 'weekend', 'onpromotion']:
            inputs[feature] = st.selectbox(f"{feature}", [0, 1])
        else:
            inputs[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    model = load_model()
    input_df = prepare_input(inputs)
    prediction = predict(model, input_df)
    st.success(f"Predicted Unit Sales: {prediction[0]:.2f}")
