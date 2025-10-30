# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import traceback

# --- Load Model, Scaler, and Feature Columns Safely ---
try:
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

except Exception as e:
    st.error("❌ Error loading model/scaler/feature columns.")
    st.text(traceback.format_exc())
    st.stop()

# --- Streamlit App ---
st.set_page_config(page_title="Manufacturing Output Predictor", page_icon="🏭", layout="wide")
st.title("🏭 Manufacturing Equipment Output Prediction App")
st.write("Predict hourly Parts Per Hour based on machine operating parameters.")

# --- User Input ---
def user_input_features():
    Injection_Temperature = st.number_input("Injection Temperature (°C)", 150.0, 300.0, 200.0)
    Injection_Pressure = st.number_input("Injection Pressure (MPa)", 50.0, 150.0, 90.0)
    Cycle_Time = st.number_input("Cycle Time (s)", 10.0, 60.0, 30.0)
    Cooling_Time = st.number_input("Cooling Time (s)", 5.0, 30.0, 15.0)
    Material_Viscosity = st.number_input("Material Viscosity", 0.5, 2.5, 1.2)
    Ambient_Temperature = st.number_input("Ambient Temperature (°C)", 20.0, 40.0, 25.0)
    Machine_Age = st.number_input("Machine Age (Years)", 0.0, 15.0, 5.0)
    Operator_Experience = st.number_input("Operator Experience (Years)", 0.0, 20.0, 5.0)
    Maintenance_Hours = st.number_input("Maintenance Hours per Month", 0.0, 50.0, 10.0)
    Temperature_Pressure_Ratio = st.number_input("Temperature/Pressure Ratio", 0.1, 5.0, 2.0)
    Total_Cycle_Time = st.number_input("Total Cycle Time (s)", 10.0, 100.0, 45.0)
    Efficiency_Score = st.number_input("Efficiency Score", 0.0, 1.0, 0.8)
    Machine_Utilization = st.number_input("Machine Utilization (%)", 0.0, 100.0, 80.0)

    Shift = st.selectbox("Shift", ["Morning", "Evening", "Night"])
    Machine_Type = st.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
    Material_Grade = st.selectbox("Material Grade", ["Grade 1", "Grade 2", "Grade 3"])
    Day_of_Week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    shift_dict = {"Morning": 0, "Evening": 1, "Night": 2}
    machine_type_dict = {"Type A": 0, "Type B": 1, "Type C": 2}
    material_grade_dict = {"Grade 1": 0, "Grade 2": 1, "Grade 3": 2}
    day_dict = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    data = {
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
        "Machine_Utilization": Machine_Utilization,
        "Shift": shift_dict[Shift],
        "Machine_Type": machine_type_dict[Machine_Type],
        "Material_Grade": material_grade_dict[Material_Grade],
        "Day_of_Week": day_dict[Day_of_Week],
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction ---
if st.button("🚀 Predict Output"):
    try:
        # Align features with training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)

        st.success(f"✅ Predicted Hourly Output (Parts Per Hour): **{prediction[0]:.2f}**")

    except Exception as e:
        st.error("❌ An error occurred during prediction.")
        st.text(traceback.format_exc())
