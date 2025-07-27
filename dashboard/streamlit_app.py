import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from gtts import gTTS

st.set_page_config(page_title="Healthcare Risk Prediction", layout="centered")
st.title("🩺 Disease Risk Prediction Dashboard")

# Load model and encoders
try:
    model = joblib.load('models/healthcare_risk_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
except Exception as e:
    st.error(f"❌ Model or encoder files not found: {e}")
    st.stop()

# Load hospital data
hospital_df = pd.read_csv('data/hospital_kpi_sample.csv') if os.path.exists('data/hospital_kpi_sample.csv') else pd.DataFrame()

def generate_audio_advice(text):
    lang_code = "en"
    tts = gTTS(text=text, lang=lang_code)
    tts.save("advice.mp3")
    with open("advice.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")

def encode_input(age, glucose, bp, bmi, insulin, pedigree, gender, smoking, exercise, history):
    try:
        gender = encoders['Gender'].transform([gender])[0]
        smoking = encoders['Smoking'].transform([smoking])[0]

        if exercise not in encoders['Exercise'].classes_:
            exercise = encoders['Exercise'].classes_[0]
        exercise = encoders['Exercise'].transform([exercise])[0]

        history = encoders['FamilyHistory'].transform([history])[0]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    return np.array([[age, gender, glucose, bp, bmi, insulin, pedigree,
                      smoking, exercise, history]])

# 🧪 Single Patient Prediction
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

if st.button("🩺 Predict Now"):
    try:
        risk = model.predict_proba(features)[0][1]
        st.metric("🧠 Risk Score", f"{risk * 100:.2f}%")
        st.progress(int(risk * 100))

        if risk >= 0.75:
            advice = (
                "You are at very high risk of developing diabetes.\n\n"
                "Medical Recommendation:\n\n"
                "- Visit an endocrinologist within 48 hours.\n"
                "- Get HbA1c, Fasting Sugar, Lipid Profile, Kidney Function Test.\n"
                "- Monitor blood sugar levels daily.\n\n"
                "- Prescription: Metformin, Insulin (based on reports).\n"
                "- Start a strict low-carb diet.\n\n"
                "- Avoid sugary foods, cold drinks, and processed snacks.\n"
                "- Estimated Cost: ₹3000 to ₹6000/month"
            )
            st.error("🔴 Very High Risk - Immediate Medical Attention Needed")

        elif risk >= 0.5:
            advice = (
                "You are at moderate risk of developing diabetes.\n\n"
                "- Schedule a checkup within 7 days.\n"
                "- Recommended: Fasting Sugar, BP Monitoring.\n"
                "- Begin daily exercise (30 min).\n"
                "- Diet: Low glycemic index.\n"
                "- Estimated Cost: ₹1000 to ₹2500/month"
            )
            st.info("🟠 Moderate Risk - Take Action Now")

        else:
            advice = (
                "Your risk is low. Keep up the good lifestyle!\n\n"
                "- Eat healthy, exercise regularly.\n"
                "- Get annual checkups.\n"
                "- Sleep 7-8 hrs, avoid sugary snacks late night.\n"
                "- Maintenance Cost: ₹300 to ₹800/year"
            )
            st.success("🟢 Low Risk - Keep Maintaining Your Health")

        # Additional doctor-like personalized checks
        doctor_notes = []

        if bp < 90:
            doctor_notes.append("🩸 Low BP: Increase salt intake and hydration.")
        elif bp > 140:
            doctor_notes.append("🩺 High BP: Reduce salt and avoid stress. Consider medication.")

        if glucose < 70:
            doctor_notes.append("🍭 Glucose too low. Eat sugar immediately and consult doctor.")
        elif glucose > 140:
            doctor_notes.append("🍬 High glucose level. Limit sugar and monitor regularly.")

        if bmi < 18.5:
            doctor_notes.append("⚖️ Underweight: Increase healthy calories, check for malnutrition.")
        elif bmi > 25:
            doctor_notes.append("📈 Overweight: Start calorie-deficit diet and light cardio.")

        if insulin < 15:
            doctor_notes.append("💉 Low insulin. Could indicate impaired beta cell function.")
        elif insulin > 150:
            doctor_notes.append("⚠️ High insulin: Risk of insulin resistance. Reduce carbs.")

        if age > 60:
            doctor_notes.append("👴 Age 60+: Annual checkups are strongly recommended.")

        if history == "Yes":
            doctor_notes.append("🧬 Family history present. Stay consistent with screenings.")

        if pedigree > 0.8:
            doctor_notes.append("📊 High diabetes pedigree score. Be extra cautious with lifestyle.")

        for note in doctor_notes:
            st.info(note)

        st.markdown("### 👨‍⚕️ Doctor's Detailed Advice")
        st.markdown(advice)
        generate_audio_advice(advice)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# 🏥 Hospital KPI Insights
if not hospital_df.empty:
    st.subheader("🏥 Hospital KPI Insights")
    selected_hospital = st.selectbox("Select a Hospital", hospital_df['HospitalName'].unique())
    kpi = hospital_df[hospital_df['HospitalName'] == selected_hospital]
    st.dataframe(kpi.drop_duplicates(subset='HospitalName').T)
else:
    st.info("Upload 'hospital_kpi_sample.csv' in data/ folder to enable KPI section.")

# 📳 Upload + Analyze Hospital KPI CSV
st.subheader("📄 Upload any Hospital KPI CSV")
hospital_upload = st.file_uploader("Upload CSV", type=["csv"], key="hospital_kpi_csv")

if hospital_upload:
    try:
        df = pd.read_csv(hospital_upload)
        st.subheader("📌 Preview")
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