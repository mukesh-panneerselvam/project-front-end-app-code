# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

# Load model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
  
st.title("🏠 House Price Prediction App")

# Input form
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"🏷️ Estimated House Price: ${prediction[0]:,.2f}")
