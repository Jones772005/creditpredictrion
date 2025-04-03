import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/record.csv")

# Selecting relevant columns
df = df[["DAYS_BIRTH", "AMT_INCOME_TOTAL", "NAME_HOUSING_TYPE", "CNT_FAM_MEMBERS", "NAME_INCOME_TYPE", "approved"]]

# Convert DAYS_BIRTH to Age
df["AGE"] = abs(df["DAYS_BIRTH"]) // 365

# Mapping categorical features
housing_mapping = {
    "Rented apartment": 0,
    "House / apartment": 1,
    "Municipal apartment": 2,
    "Co-op apartment": 3,
    "Office apartment": 4
}

job_mapping = {
    "Working": 0,
    "Commercial associate": 1,
    "Pensioner": 2,
    "State servant": 3,
    "Unemployed": 4
}

df["NAME_HOUSING_TYPE"] = df["NAME_HOUSING_TYPE"].map(housing_mapping)
df["NAME_INCOME_TYPE"] = df["NAME_INCOME_TYPE"].map(job_mapping)

# Features & Labels
X = df[["AGE", "AMT_INCOME_TOTAL", "NAME_HOUSING_TYPE", "CNT_FAM_MEMBERS", "NAME_INCOME_TYPE"]]
y = df["approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model & scaler properly
with open("model/best_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("model/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and Scaler saved successfully!")
