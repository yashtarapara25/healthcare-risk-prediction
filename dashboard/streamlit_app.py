
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import io
from gtts import gTTS

st.set_page_config(page_title="Healthcare Risk Prediction", layout="centered")
st.title("🩺 Disease Risk Prediction Dashboard")

# Load model
# Load model
try:
    model = joblib.load('models/disease_predictor.pkl')  # ✅ fixed path
except:
    st.error("❌ Model file not found! Make sure 'disease_predictor.pkl' is in the models folder.")
    st.stop()

# Load hospital data
hospital_df = pd.read_csv('data/hospital_kpi_sample.csv') if os.path.exists('data/hospital_kpi_sample.csv') else pd.DataFrame()

# Text-to-speech and video generation (without moviepy becasue in python 13.3 in moviepy not supported)  
def generate_video_advice(text):
    lang_code = "en"
    tts = gTTS(text=text, lang=lang_code)
    tts.save("advice.mp3")

    image_file = os.path.join("dashboard", "doctor_image.jpg")

    video_file = "advice_video.mp4"

    if os.path.exists(image_file):
        ffmpeg_command = (
            f'ffmpeg -y -loop 1 -i "{image_file}" -i "advice.mp3" '
            f'-c:v libx264 -tune stillimage -c:a aac -b:a 192k '
            f'-pix_fmt yuv420p -shortest "{video_file}"'
        )
        os.system(ffmpeg_command)

        if os.path.exists(video_file):
            with open(video_file, "rb") as vf:
                st.video(vf.read())
        else:
            st.error("❌ Video file not generated.")
    else:
        with open("advice.mp3", "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")


# Encode patient input
def encode_input(age, glucose, bp, bmi, insulin, pedigree, gender, smoking, exercise, history):
    gender_male = 1 if gender == "Male" else 0
    smoking_yes = 1 if smoking == "Yes" else 0
    exercise_regular = 1 if exercise == "Regular" else 0
    history_yes = 1 if history == "Yes" else 0
    return [age, glucose, bp, bmi, insulin, pedigree, gender_male, smoking_yes, exercise_regular, history_yes]

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

if st.button("🧪 Predict Now"):
    risk = model.predict_proba([features])[0][1]
    st.metric("🧠 Risk Score", f"{risk * 100:.2f}%")
    st.progress(int(risk * 100))

    if risk >= 0.75:
        advice = (
            " You are at very high risk of developing diabetes.\n\n"
            " Medical Recommendation:\n\n"
            "- Visit an endocrinologist within 48 hours.\n"
            "- Get the following tests done: HbA1c, Fasting Sugar, Lipid Profile, Kidney Function Test.\n"
            "- Monitor your blood sugar levels daily.\n\n"
            "- Treatment Plan:\n"
            "- Likely prescription: Metformin, Insulin (based on test reports).\n"
            "- Start a strict low-carb diet.\n\n"
            "- Additional Advice:\n\n"
            "- Avoid sugary foods, cold drinks, and processed snacks.\n"
            "- Check for foot numbness or blurred vision.\n\n"
            "- Estimated Cost: ₹3000 to ₹6000/month"
        )
        st.error("🔴 Very High Risk - Immediate Medical Attention Needed")

    elif risk >= 0.5:
        advice = (
            " You are at moderate risk of developing diabetes.\n\n"
            " Medical Recommendation:\n\n"
            "- Schedule a checkup within the next 7 days.\n\n"
            "- Recommended tests: Fasting Blood Sugar, BP Monitoring.\n\n"
            "- Begin a regular exercise routine (at least 30 min daily).\n\n"
            "- Suggested Lifestyle/Treatment:\n\n"
            "- Follow a low glycemic index diet.\n\n"
            "- Take prescribed medicine like Glimepiride if suggested by doctor.\n\n"
            "- Estimated Cost: ₹1000 to ₹2500/month"
        )
        st.info("🟠 Moderate Risk - Take Action Now")

    else:
        advice = (
            " Your risk is low, and your current lifestyle is keeping you healthy.\n\n"
            " Recommended Care:\n"
            "Continue healthy eating and regular exercise.\n\n"
            "Get annual checkups: Fasting Blood Sugar, BP.\n\n"
            " Pro Tips:\n"
            " Sleep 7-8 hours daily.\n"
            " Avoid late night meals and sugary snacks.\n\n"
            " Estimated Maintenance Cost: ₹300 to ₹800/year"
        )
        st.success("🟢 Low Risk - Keep Maintaining Your Health")

    st.markdown("### 👨‍⚕️ Doctor's Detailed Advice")
    st.markdown(advice)
    generate_video_advice(advice)

# 🏥 Hospital KPI Insights
if not hospital_df.empty:
    st.subheader("🏥 Hospital KPI Insights")
    selected_hospital = st.selectbox("Select a Hospital", hospital_df['HospitalName'].unique())
    kpi = hospital_df[hospital_df['HospitalName'] == selected_hospital]
    st.dataframe(kpi.drop_duplicates(subset='HospitalName').T)
else:
    st.info("Upload 'hospital_kpi_sample.csv' in data/ folder to enable KPI section.")

# 🧾 Upload + Analyze Hospital KPI CSV
st.subheader("📤 Upload any Hospital KPI CSV")
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
        st.error(f"❌ Error: {e}")
