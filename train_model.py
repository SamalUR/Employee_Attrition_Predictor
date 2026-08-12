import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_processing import load_and_preprocess_data

def train_and_evaluate():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data('HR-Employee-Attrition.csv')

    # Models Define
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    best_model = None
    best_f1 = 0
    best_model_name = ""

    # Models Train and Evaluation
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 2),
            'Precision': round(prec, 2),
            'Recall': round(rec, 2),
            'F1-Score': round(f1, 2),
            'ROC-AUC': round(auc, 2)
        })

        # F1-Score 
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    # Results Dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_comparison.csv', index=False)
    
    # Best Model
    joblib.dump(best_model, 'best_model.pkl')

    print("📊 Model Comparison Results:")
    print(results_df)
    print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.2f}) - 'best_model.pkl' ලෙස Save විය.")

if __name__ == "__main__":
    train_and_evaluate()