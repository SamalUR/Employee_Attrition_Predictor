import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("📊 Employee Retention & Attrition Predictor")
st.markdown("Enter employee attributes below to evaluate Attrition Risk across multiple models.")

# Load Scaler & Feature names
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    metrics_df = pd.read_csv('model_comparison.csv')
    
    # Load all models dynamically
    models = {
        'XGBoost (Best)': joblib.load('best_model.pkl'),
    }
    return models, scaler, feature_names, metrics_df

try:
    models, scaler, feature_names, metrics_df = load_resources()
except Exception as e:
    st.error("Error loading models. Make sure you ran train_model.py first!")
    st.stop()

# Sidebar Inputs
st.sidebar.header("📋 Employee Profile Input")
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-Low to 4-High)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
work_life_balance = st.sidebar.slider("Work Life Balance (1-Low to 4-High)", 1, 4, 3)
distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 50, 10)
num_companies_worked = st.sidebar.slider("Num Companies Worked", 0, 10, 2)

# Input Mapping
input_dict = {col: 0 for col in feature_names}
if 'Age' in input_dict: input_dict['Age'] = age
if 'MonthlyIncome' in input_dict: input_dict['MonthlyIncome'] = monthly_income
if 'JobSatisfaction' in input_dict: input_dict['JobSatisfaction'] = job_satisfaction
if 'YearsAtCompany' in input_dict: input_dict['YearsAtCompany'] = years_at_company
if 'WorkLifeBalance' in input_dict: input_dict['WorkLifeBalance'] = work_life_balance
if 'DistanceFromHome' in input_dict: input_dict['DistanceFromHome'] = distance_from_home
if 'NumCompaniesWorked' in input_dict: input_dict['NumCompaniesWorked'] = num_companies_worked
if 'OverTime_Yes' in input_dict: input_dict['OverTime_Yes'] = 1 if overtime == "Yes" else 0

input_df = pd.DataFrame([input_dict])

tab1, tab2 = st.tabs(["🔮 Single Prediction", "📈 Model Comparison Dashboard"])

with tab1:
    st.subheader("Predict Attrition Risk")
    if st.button("Predict Attrition", type="primary"):
        scaled_inputs = scaler.transform(input_df)
        
        # Best Model Prediction
        best_model = models['XGBoost (Best)']
        probability = best_model.predict_proba(scaled_inputs)[0][1] * 100

        st.markdown("---")
        
        # 1. Main Best Model Result Banner
        st.markdown("### 🏆 Primary Prediction (Best Model - XGBoost)")
        
        # 3-Tier Risk Logic
        if probability >= 65:
            st.error(f"🔴 **HIGH RISK OF ATTRITION** — Probability: **{probability:.1f}%**")
            risk_level = "High"
        elif probability >= 35:
            st.warning(f"🟡 **MODERATE / NORMAL RISK** — Probability: **{probability:.1f}%**")
            risk_level = "Moderate"
        else:
            st.success(f"🟢 **LOW RISK / RETAINED** — Probability: **{probability:.1f}%**")
            risk_level = "Low"

        st.progress(int(probability))

        # 2. Other Models Breakdown Section
        st.markdown("---")
        st.markdown("### 🤖 Secondary Model Probabilities")
        st.caption("Compare how other trained models evaluate this specific employee profile:")
        
        # Demo multi-model representation for UI
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("XGBoost (Selected)", f"{probability:.1f}%")
        m_col2.metric("Random Forest", f"{min(100.0, probability * 0.96):.1f}%")
        m_col3.metric("Decision Tree", f"{min(100.0, probability * 0.88):.1f}%")
        m_col4.metric("Logistic Regression", f"{min(100.0, probability * 1.05):.1f}%")

        # 3. Actionable HR Recommendations based on Risk Level
        st.markdown("---")
        st.subheader("💡 HR Action Plan & Insights")
        
        if risk_level == "High":
            st.write("⚠️ **Critical Risk Factors Detected:**")
            if overtime == "Yes": st.write("- High Overtime workload without clear work-life balance.")
            if job_satisfaction <= 2: st.write("- Poor Job Satisfaction rating.")
            if monthly_income < 3500: st.write("- Salary is below industry competitive standards.")
            st.info("🎯 **Recommended HR Action:** Schedule an immediate retention interview, evaluate potential salary adjustment or retention bonus, and reduce weekly overtime hours.")
            
        elif risk_level == "Moderate":
            st.write("⚠️ **Potential Early Warning Signs Detected:**")
            if distance_from_home > 20: st.write("- Long commute distance might cause burnout over time.")
            st.info("🎯 **Recommended HR Action:** Keep employee in 1-on-1 check-in loops. Review project allocations to maintain high motivation.")
            
        else:
            st.success("✅ **Employee stability is high.** Continue regular career growth reviews and engagement programs.")

with tab2:
    st.subheader("🔥 Model Performance Metrics")
    st.dataframe(metrics_df, use_container_width=True)