import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from gtts import gTTS

# Page setup
st.set_page_config(page_title="Healthcare Risk Prediction", layout="centered")
st.title("🩺 Disease Risk Prediction Dashboard")

# Load model
try:
    model = joblib.load('models/disease_predictor.pkl')
except FileNotFoundError:
    st.error("❌ Model file not found! Ensure 'disease_predictor.pkl' is inside the 'models/' folder.")
    st.stop()

# Load hospital data
hospital_df = pd.read_csv('data/hospital_kpi_sample.csv') if os.path.exists('data/hospital_kpi_sample.csv') else pd.DataFrame()

# Function: Text-to-speech audio only
def generate_audio_advice(text):
    lang_code = "en"
    tts = gTTS(text=text, lang=lang_code)
    tts.save("advice.mp3")
    with open("advice.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")

# Function: Encode user input
def encode_input(age, glucose, bp, bmi, insulin, pedigree, gender, smoking, exercise, history):
    gender_male = 1 if gender == "Male" else 0
    smoking_yes = 1 if smoking == "Yes" else 0
    exercise_regular = 1 if exercise == "Regular" else 0
    history_yes = 1 if history == "Yes" else 0
    return [age, glucose, bp, bmi, insulin, pedigree, gender_male, smoking_yes, exercise_regular, history_yes]

# ==========================
# 🔍 SINGLE PATIENT PREDICTION
# ==========================
st.subheader("🔍 Predict Single Patient Risk")

age = st.slider("Age", 18, 100, 30)
glucose = st.slider("Glucose", 50, 200, 100)
bp = st.slider("Blood Pressure", 40, 150, 90)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
insulin = st.slider("Insulin", 0, 300, 80)
pedigree = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
exercise = st.selectbox("Exercise", ["Regular", "Occasional", "None"])
history = st.selectbox("Family History", ["Yes", "No"])

features = encode_input(age, glucose, bp, bmi, insulin, pedigree, gender, smoking, exercise, history)

if st.button("🧪 Predict Now"):
    risk = model.predict_proba([features])[0][1]
    st.metric("🧠 Risk Score", f"{risk * 100:.2f}%")
    st.progress(int(risk * 100))

    if risk >= 0.75:
        advice = (
            "You are at very high risk of developing diabetes.\n"
            "- Visit an endocrinologist within 48 hours.\n"
            "- Get tests: HbA1c, Lipid Profile, Kidney Function.\n"
            "- Likely medicines: Metformin, Insulin.\n"
            "- Estimated Cost: ₹3000–₹6000/month."
        )
        st.error("🔴 Very High Risk - Immediate Medical Attention Needed")
    elif risk >= 0.5:
        advice = (
            "You are at moderate risk of developing diabetes.\n"
            "- Schedule a checkup within 7 days.\n"
            "- Begin exercise and monitor sugar levels.\n"
            "- Possible medicine: Glimepiride.\n"
            "- Estimated Cost: ₹1000–₹2500/month."
        )
        st.info("🟠 Moderate Risk - Take Action Now")
    else:
        advice = (
            "You are at low risk. Keep up your healthy lifestyle.\n"
            "- Maintain healthy diet & regular exercise.\n"
            "- Get annual checkups (Fasting Blood Sugar, BP).\n"
            "- Estimated Maintenance Cost: ₹300–₹800/year."
        )
        st.success("🟢 Low Risk - Keep Maintaining Your Health")

    st.markdown("### 👨‍⚕️ Doctor's Detailed Advice")
    st.markdown(advice)
    generate_audio_advice(advice)

# ==========================
# 🏥 HOSPITAL KPI INSIGHTS
# ==========================
if not hospital_df.empty:
    st.subheader("🏥 Hospital KPI Insights")
    selected_hospital = st.selectbox("Select a Hospital", hospital_df['HospitalName'].unique())
    kpi = hospital_df[hospital_df['HospitalName'] == selected_hospital]
    st.dataframe(kpi.drop_duplicates(subset='HospitalName').T)
else:
    st.info("Upload 'hospital_kpi_sample.csv' into 'data/' folder to view KPIs.")

# ==========================
# 📤 CSV UPLOAD FOR ANALYSIS
# ==========================
st.subheader("📤 Upload any Hospital KPI CSV")
hospital_upload = st.file_uploader("Upload CSV", type=["csv"], key="hospital_kpi_csv")

if hospital_upload:
    try:
        df = pd.read_csv(hospital_upload)
        st.subheader("📌 Data Preview")
        st.dataframe(df.head())

        st.subheader("📊 Descriptive Statistics")
        numeric_cols = df.select_dtypes(include='number')
        if not numeric_cols.empty:
            summary_stats = numeric_cols.describe().T.round(2)
            st.dataframe(summary_stats)
            st.download_button("⬇️ Download Summary CSV", summary_stats.to_csv().encode('utf-8'), "summary.csv", "text/csv")

            st.subheader("📉 Max vs Min vs Difference")
            comparison = pd.DataFrame({
                "Max": numeric_cols.max(),
                "Min": numeric_cols.min(),
                "Difference": numeric_cols.max() - numeric_cols.min()
            }).round(2)
            st.dataframe(comparison)

        if 'HospitalName' in df.columns:
            st.subheader("🏆 Top Hospital Rankings")

            if 'TreatmentSuccessRate' in df.columns:
                top = df.groupby('HospitalName')['TreatmentSuccessRate'].mean().sort_values(ascending=False).head(5)
                st.dataframe(top.reset_index())
            if 'AvgStayDays' in df.columns:
                stay = df.groupby('HospitalName')['AvgStayDays'].mean().sort_values(ascending=False).head(5)
                st.dataframe(stay.reset_index())
            if 'ReadmissionRate' in df.columns:
                readmit = df.groupby('HospitalName')['ReadmissionRate'].mean().sort_values().head(5)
                st.dataframe(readmit.reset_index())
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
