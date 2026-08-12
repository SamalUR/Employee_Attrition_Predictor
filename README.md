# Employee Retention & Attrition Predictor

An End-to-End Machine Learning web application designed to predict employee attrition risk, compare multiple classification models, and provide actionable HR decision insights to reduce employee turnover.

---
<img width="1874" height="868" alt="Screenshot_12-8-2026_224619_localhost" src="https://github.com/user-attachments/assets/9052c302-76e5-4950-a451-e21cddcc9957" />



## Key Features

* **3-Tier Attrition Risk Categorization:** Categorizes risk into **Low Risk (0-35%)**, **Moderate Risk %)**, and **High Risk (65-100%)** for practical HR decision-making.
* **Multi-Model Comparison Dashboard:** Evaluates **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost Classifier** using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Class Imbalance Handling:** Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle dataset class imbalance (Attrition Yes vs. No).
* **Dynamic HR Recommendations:** Automatically flags top individual risk factors (e.g., High OverTime, Low Job Satisfaction, Low Monthly Income) and suggests targeted retention actions.
* **Interactive Web UI:** Built using **Streamlit** for seamless interaction and real-time inference.

---

## Project Architecture & Tech Stack

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), XGBoost
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit
* **Dataset:** IBM HR Analytics Employee Attrition & Performance (Kaggle)

---

## Repository Folder Structure
Employee_Attrition_Predictor/
│
├── HR-Employee-Attrition.csv                # Kaggle Dataset
├── data_processing.py                       # Data Cleaning, Encoding & SMOTE Balancing
├── train_model.py                           # Model Training, Evaluation & Saving Best Model
├── app.py                                   # Streamlit Web Application (Dashboard)
├── requirements.txt                         # Project Dependencies
└── README.md                                # Project Documentation

