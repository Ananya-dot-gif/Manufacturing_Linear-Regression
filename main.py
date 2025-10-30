# main.py

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n1. Project Title: Data Science Capstone Project - Manufacturing Equipment Output Prediction with Linear Regression")
print("\n2. Category: Supervised Learning (Regression)")
print("""
3. Problem Statement:
Predict the hourly output (number of parts produced per hour) based on various machine parameters like temperature, pressure, and material properties.
""")

# Load Dataset
df = pd.read_csv(r"D:\MUFG\manufacturing_dataset_1000_samples project1 Capstone.csv")

# Drop unnecessary columns
if "Timestamp" in df.columns:
    df.drop("Timestamp", axis=1, inplace=True)

# Fill missing values
df["Material_Viscosity"] = df["Material_Viscosity"].fillna(df["Material_Viscosity"].mean())
df["Ambient_Temperature"] = df["Ambient_Temperature"].fillna(df["Ambient_Temperature"].mean())
df["Operator_Experience"] = df["Operator_Experience"].fillna(df["Operator_Experience"].mean())

# Encode categorical columns
le = LabelEncoder()
df["Shift"] = le.fit_transform(df["Shift"])
df["Machine_Type"] = le.fit_transform(df["Machine_Type"])
df["Material_Grade"] = le.fit_transform(df["Material_Grade"])
df["Day_of_Week"] = le.fit_transform(df["Day_of_Week"])

# Features & Target
X = df.drop("Parts_Per_Hour", axis=1)
y = df["Parts_Per_Hour"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

# Save model, scaler, and columns
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("\n✅ Model, Scaler, and Feature Columns saved successfully!")
