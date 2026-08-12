import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    unnecessary_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=[col for col in unnecessary_cols if col in df.columns])

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, feature_names

if __name__ == "__main__":
    # Locate a likely Attrition CSV in the project root
    possible_files = [
        'HR-Employee-Attrition.csv',
        'Attrition.csv',
        'HR_Employee_Attrition.csv'
    ]
    csv_path = None
    for p in possible_files:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            'Could not find an Attrition CSV in the project root. '
            'Place the dataset in the project folder or pass a filepath.'
        )

    X_train, X_test, y_train, y_test, features = load_and_preprocess_data(csv_path)
    print("✅ Data Preprocessing Completed")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")