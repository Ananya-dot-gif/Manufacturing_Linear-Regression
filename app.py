# ------------------------------------------------------------
# app.py — Modern Streamlit App for Manufacturing Output Prediction
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Page Configuration
st.set_page_config(
    page_title="🏭 Manufacturing Output Predictor",
    page_icon="⚙️",
    layout="centered",
)

# ------------------------------------------------------------
# Custom CSS for modern look
st.markdown("""
    <style>
        .main {
            background-color: #fdf6f0;
        }
        .stButton>button {
            background-color: #ff7b54;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #e36443;
        }
        .prediction-box {
            background-color: #fffaf0;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #ffb677;
            text-align: center;
            box-shadow: 0px 0px 10px rgba(255,180,100,0.3);
        }
        .title {
            text-align: center;
            color: #ff7b54;
            font-size: 28px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header
st.markdown("<div class='title'>🏭 Manufacturing Output Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict hourly machine output using smart linear regression</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------
# Check for model files
required_files = ['model.pkl', 'scaler.pkl', 'feature_names.pkl']
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"⚠️ Missing files: {', '.join(missing)}")
    st.info("Please run **main.py** first to train and generate model files.")
    st.stop()

# ------------------------------------------------------------
# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# ------------------------------------------------------------
# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1995/1995574.png", width=100)
st.sidebar.title("⚙️ Menu")
page = st.sidebar.radio("Navigate", ["🔮 Prediction", "📊 About Project", "👩‍💻 Developer Info"])

# ------------------------------------------------------------
# 🔮 Prediction Page
if page == "🔮 Prediction":
    st.subheader("Enter Machine Parameters Below 👇")

    # Create columns for neat layout
    input_data = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        with (col1 if i % 2 == 0 else col2):
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([input_data])

    if st.button("🚀 Predict Output"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        st.markdown(
            f"""
            <div class='prediction-box'>
                <h3>✅ Predicted Output</h3>
                <h1 style='color:#ff7b54;'>{prediction[0]:.2f} Parts/Hour</h1>
                <p>This is the expected hourly machine performance.</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 📊 About Project Page
elif page == "📊 About Project":
    st.subheader("📘 Project Overview")
    st.write("""
    This project uses **Linear Regression** to predict the number of parts produced per hour
    based on various manufacturing parameters such as temperature, pressure, cycle time,
    and material properties.

    The goal is to help optimize:
    - ⚙️ Machine settings  
    - 🧾 Production scheduling  
    - 🚨 Detection of under-performing machines  
    """)

    st.success("Technologies Used: Python · Streamlit · Scikit-Learn · Pandas · NumPy")

# ------------------------------------------------------------
# 👩‍💻 Developer Info Page
elif page == "👩‍💻 Developer Info":
    st.subheader("About the Developer")
    st.write("""
    **👩‍🎓 Name:** Ananya S Bharadwaj  
    **🎯 Role:** AI/ML Engineer | Capstone Project Developer  
    **💡 Goal:** To build intelligent systems that solve real-world industrial challenges.  
    """)
    st.markdown("🔗 *Built with ❤️ using Streamlit*")

