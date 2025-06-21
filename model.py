import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# 🔁 Replace with your actual CSV file name
data = pd.read_csv("crop_yield.csv")  # Must include 'Yield' column

# ✅ Define features and label
X = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define column types
categorical_cols = ['Crop', 'Season', 'State']
numerical_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# ✅ Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# ✅ Full pipeline with DecisionTreeRegressor
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])

# ✅ Train model
pipeline.fit(X_train, y_train)

# ✅ Extract model and preprocessor
model = pipeline.named_steps['model']
preprocess = pipeline.named_steps['preprocess']

# ✅ Save model and preprocessor
with open("dtr.pkl", "wb") as f:
    pickle.dump(model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocess, f)

print("✅ Model and preprocessor saved successfully using scikit-learn 1.7.0")
