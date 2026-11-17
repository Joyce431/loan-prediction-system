import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("--- 1. Loading Data ---")
# Load data (make sure your CSV is named this exactly)
df = pd.read_csv("data/loan_data_set.csv")

print("--- 2. Cleaning & Feature Engineering ---")
# Drop ID
df = df.drop("Loan_ID", axis=1)

# Fix Dependents
df['Dependents'] = df['Dependents'].replace('3+', '3')

# Fill missing values
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())
    
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature Engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
df['Balance_Income'] = df['TotalIncome'] - df['EMI']

# Target mapping
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- 3. Building Pipeline ---")
num_features = ['Dependents', 'LoanAmount', 'Credit_History', 'TotalIncome', 'EMI', 'Balance_Income']
cat_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

print("--- 4. Training Models ---")
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

best_f1 = 0
best_model_name = ""
best_pipeline = None

for name, model in models.items():
    print(f"Training {name}...")
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1 Score: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = clf

print(f"\n--- 5. Saving Best Model ({best_model_name}) ---")
# Save the model for the streamlit app
joblib.dump(best_pipeline, "models/best_model.joblib")
# Also save as .pkl just in case
joblib.dump(best_pipeline, "models/best_model.pkl") 
joblib.dump(best_pipeline, "loan_model.pkl") # Save in root folder too
print("Model saved successfully!")
print("Done! You can now upload the 'models' folder and the 'loan_model.pkl' file to GitHub.")