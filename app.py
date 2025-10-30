import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ------------------------------
# Load Model, Scaler, and Columns
# ------------------------------
@st.cache_resource
def load_model_and_scaler():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_model_and_scaler()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Manufacturing Output Predictor", page_icon="🏭", layout="centered")

st.title("🏭 Manufacturing Equipment Output Prediction")
st.markdown("""
### Data Science Capstone Project: Predicting Hourly Machine Output using Linear Regression
Optimize manufacturing efficiency by predicting parts produced per hour based on machine operating parameters.
""")

st.sidebar.header("Input Machine Parameters")

# ------------------------------
# Input Fields
# ------------------------------

Shift = st.sidebar.selectbox("Shift", ["Morning", "Evening", "Night"])
Machine_Type = st.sidebar.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
Material_Grade = st.sidebar.selectbox("Material Grade", ["High", "Medium", "Low"])
Day_of_Week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

Injection_Temperature = st.sidebar.number_input("Injection Temperature", min_value=100.0, max_value=400.0, value=250.0)
Injection_Pressure = st.sidebar.number_input("Injection Pressure", min_value=10.0, max_value=300.0, value=120.0)
Cycle_Time = st.sidebar.number_input("Cycle Time", min_value=1.0, max_value=100.0, value=30.0)
Cooling_Time = st.sidebar.number_input("Cooling Time", min_value=1.0, max_value=50.0, value=15.0)
Material_Viscosity = st.sidebar.number_input("Material Viscosity", min_value=0.1, max_value=10.0, value=5.0)
Ambient_Temperature = st.sidebar.number_input("Ambient Temperature", min_value=10.0, max_value=50.0, value=25.0)
Machine_Age = st.sidebar.number_input("Machine Age (years)", min_value=0.0, max_value=20.0, value=5.0)
Operator_Experience = st.sidebar.number_input("Operator Experience (years)", min_value=0.0, max_value=20.0, value=3.0)
Maintenance_Hours = st.sidebar.number_input("Maintenance Hours", min_value=0.0, max_value=100.0, value=20.0)
Temperature_Pressure_Ratio = st.sidebar.number_input("Temperature Pressure Ratio", min_value=0.0, max_value=10.0, value=2.5)
Total_Cycle_Time = st.sidebar.number_input("Total Cycle Time", min_value=1.0, max_value=200.0, value=45.0)
Efficiency_Score = st.sidebar.number_input("Efficiency Score", min_value=0.0, max_value=1.0, value=0.8)
Machine_Utilization = st.sidebar.number_input("Machine Utilization (%)", min_value=0.0, max_value=1.0, value=0.75)

# ------------------------------
# Encode Categorical Inputs
# ------------------------------
label_encoders = {
    "Shift": {"Morning": 2, "Evening": 0, "Night": 1},
    "Machine_Type": {"Type A": 2, "Type B": 1, "Type C": 0},
    "Material_Grade": {"High": 0, "Medium": 1, "Low": 2},
    "Day_of_Week": {"Monday": 1, "Tuesday": 6, "Wednesday": 5, "Thursday": 4, "Friday": 0, "Saturday": 3, "Sunday": 2}
}

input_data = {
    "Shift": label_encoders["Shift"][Shift],
    "Machine_Type": label_encoders["Machine_Type"][Machine_Type],
    "Material_Grade": label_encoders["Material_Grade"][Material_Grade],
    "Day_of_Week": label_encoders["Day_of_Week"][Day_of_Week],
    "Injection_Temperature": Injection_Temperature,
    "Injection_Pressure": Injection_Pressure,
    "Cycle_Time": Cycle_Time,
    "Cooling_Time": Cooling_Time,
    "Material_Viscosity": Material_Viscosity,
    "Ambient_Temperature": Ambient_Temperature,
    "Machine_Age": Machine_Age,
    "Operator_Experience": Operator_Experience,
    "Maintenance_Hours": Maintenance_Hours,
    "Temperature_Pressure_Ratio": Temperature_Pressure_Ratio,
    "Total_Cycle_Time": Total_Cycle_Time,
    "Efficiency_Score": Efficiency_Score,
    "Machine_Utilization": Machine_Utilization
}

input_df = pd.DataFrame([input_data])

# ------------------------------
# Align Columns (Critical Fix)
# ------------------------------
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ------------------------------
# Predict Output
# ------------------------------
if st.button("🔮 Predict Output"):
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.success(f"✅ Predicted Parts Per Hour: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction:\n\n{e}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
---
**Project Title:** Manufacturing Equipment Output Prediction  
**Category:** Supervised Learning (Regression)  
**Model Used:** Linear Regression  
**Developed by:** Ananya S Bharadwaj 👩‍💻
""")
