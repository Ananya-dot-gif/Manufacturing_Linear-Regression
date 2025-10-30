import streamlit as st
import pandas as pd
import pickle

# -----------------------------------------------------------
# Load the trained model
# -----------------------------------------------------------
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Manufacturing Output Prediction", page_icon="🏭", layout="wide")

# -----------------------------------------------------------
# App Title
# -----------------------------------------------------------
st.title("🏭 Manufacturing Equipment Output Prediction")
st.markdown("""
Data Science Capstone Project: **Predicting Hourly Machine Output using Linear Regression**

Optimize manufacturing efficiency by predicting parts produced per hour 
based on machine operating parameters.
""")

# -----------------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------------
st.sidebar.header("Input Machine Parameters")

shift = st.sidebar.selectbox("Shift", ["Morning", "Evening", "Night"])
machine_type = st.sidebar.selectbox("Machine Type", ["Type A", "Type B", "Type C"])
material_grade = st.sidebar.selectbox("Material Grade", ["High", "Medium", "Low"])
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

injection_temp = st.sidebar.number_input("Injection Temperature", min_value=100.0, max_value=400.0, value=250.0)
injection_pressure = st.sidebar.number_input("Injection Pressure", min_value=50.0, max_value=300.0, value=120.0)
cycle_time = st.sidebar.number_input("Cycle Time", min_value=1.0, max_value=100.0, value=20.0)
cooling_time = st.sidebar.number_input("Cooling Time", min_value=1.0, max_value=50.0, value=5.0)
operator_experience = st.sidebar.slider("Operator Experience (years)", 0, 20, 5)
ambient_temp = st.sidebar.number_input("Ambient Temperature", min_value=10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity", min_value=0.0, max_value=100.0, value=50.0)

# -----------------------------------------------------------
# Create input DataFrame
# -----------------------------------------------------------
input_dict = {
    'Shift': [shift],
    'Machine_Type': [machine_type],
    'Material_Grade': [material_grade],
    'Day_of_Week': [day_of_week],
    'Injection_Temperature': [injection_temp],
    'Injection_Pressure': [injection_pressure],
    'Cycle_Time': [cycle_time],
    'Cooling_Time': [cooling_time],
    'Operator_Experience': [operator_experience],
    'Ambient_Temperature': [ambient_temp],
    'Humidity': [humidity]
}

input_df = pd.DataFrame(input_dict)

# -----------------------------------------------------------
# One-Hot Encode Categorical Variables (same as training)
# -----------------------------------------------------------
# Define the categories in the same order used during training
shift_categories = ['Morning', 'Evening', 'Night']
machine_categories = ['Type A', 'Type B', 'Type C']
material_categories = ['High', 'Medium', 'Low']
day_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Manual encoding to ensure identical columns
for cat in shift_categories:
    input_df[f'Shift_{cat}'] = 1 if shift == cat else 0

for cat in machine_categories:
    input_df[f'Machine_Type_{cat}'] = 1 if machine_type == cat else 0

for cat in material_categories:
    input_df[f'Material_Grade_{cat}'] = 1 if material_grade == cat else 0

for cat in day_categories:
    input_df[f'Day_of_Week_{cat}'] = 1 if day_of_week == cat else 0

# Drop original categorical columns
input_df.drop(['Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week'], axis=1, inplace=True)

# -----------------------------------------------------------
# Debug display (optional - can remove later)
# -----------------------------------------------------------
st.write("🧩 Final input features sent to model:", list(input_df.columns))

# -----------------------------------------------------------
# Prediction
# -----------------------------------------------------------
if st.button("🔮 Predict Output"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Predicted Parts Produced per Hour: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction:\n\n{e}")
