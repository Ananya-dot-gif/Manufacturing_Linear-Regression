# ------------------------------------------------------------
# main.py — Train and save Linear Regression model
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# 1️⃣ Load Dataset
data = pd.read_csv("manufacturing_dataset_1000_samples project1.csv")

print("✅ Dataset loaded successfully!")

# 2️⃣ Handle Missing Values
num_cols = ['Material_Viscosity', 'Ambient_Temperature', 'Operator_Experience']
for col in num_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# 3️⃣ Identify and Encode Categorical Variables Automatically
cat_cols = data.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    print(f"Encoding categorical columns: {cat_cols}")
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
else:
    print("No categorical columns found.")

# 4️⃣ Define Features (X) and Target (y)
X = data.drop('Parts_Per_Hour', axis=1)
y = data['Parts_Per_Hour']

# 5️⃣ Ensure all columns are numeric
print("\nColumns in X after encoding:")
print(X.dtypes)

# 6️⃣ Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8️⃣ Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 9️⃣ Save Model and Scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("\n✅ Model training completed and saved successfully!")
