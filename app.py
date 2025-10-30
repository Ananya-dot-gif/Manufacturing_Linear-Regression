# ------------------------------------------------------------
# app.py — Streamlit Web Interface for Manufacturing Prediction
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Manufacturing Output Predictor", layout="centered")

# ------------------------------------------------------------
# App Title
st.title("🏭 Manufacturing Equipment Output Prediction")
st.write("Predict **Parts Per Hour** based on machine operating parameters using Linear Regression.")

# ------------------------------------------------------------
# User Input Section
st.header("Enter Machine Parameters")

# Create input fields for each feature
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# ------------------------------------------------------------
# Prediction Section
if st.button("Predict Output"):
    # Scale input
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    st.success(f"✅ Predicted Parts Per Hour: {prediction[0]:.2f}")

# ------------------------------------------------------------
# About Section
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This project demonstrates:
- Data preprocessing and Linear Regression
- Predicting machine output for optimization
- Built with ❤️ using Streamlit and Scikit-learn
""")

st.sidebar.markdown("**Developer:** Your Name")
