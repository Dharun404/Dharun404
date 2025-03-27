import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fpdf import FPDF
from PIL import Image
import os

# Load models
def load_model_file(model_path):
    return load_model(model_path)

heart_model = load_model_file('heart_disease_model.h5')
ec_model = load_model_file('ecg_model.h5')
spect_model = load_model_file('spect_model.h5')

# Load and preprocess data
data = pd.read_csv('heart.csv')
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
scaler = StandardScaler().fit(data[numerical_cols])
encoders = {col: LabelEncoder().fit(data[col]) for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']}

def preprocess_input(data):
    encoded_data = data.copy()
    for col, encoder in encoders.items():
        encoded_data[col] = encoder.transform([encoded_data[col]])[0]
    numerical_data = np.array([encoded_data[col] for col in numerical_cols]).reshape(1, -1)
    scaled_data = scaler.transform(numerical_data)
    padded_data = np.pad(scaled_data, ((0, 0), (0, 5)), 'constant')
    return padded_data

# Generate PDF report with images
def generate_pdf_report(name, age, sex, heart_pred, ecg_pred, spect_pred, ecg_image, spect_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Heart Disease Prediction Report', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, f'Name: {name}', ln=True)
    pdf.cell(200, 10, f'Age: {age}', ln=True)
    pdf.cell(200, 10, f'Sex: {sex}', ln=True)
    pdf.cell(200, 10, f'Heart Disease Prediction: {"Detected" if heart_pred > 0.5 else "Not Detected"}', ln=True)
    pdf.cell(200, 10, f'ECG Prediction: {"Abnormal" if ecg_pred > 0.5 else "Normal"}', ln=True)
    pdf.cell(200, 10, f'SPECT Prediction: {"Abnormal" if spect_pred > 0.5 else "Normal"}', ln=True)

    # Add ECG image
    if ecg_image:
        ecg_path = 'ecg_temp.png'
        ecg_image.save(ecg_path)
        pdf.image(ecg_path, x=10, y=120, w=90)
        os.remove(ecg_path)

    # Add SPECT image
    if spect_image:
        spect_path = 'spect_temp.png'
        spect_image.save(spect_path)
        pdf.image(spect_path, x=110, y=120, w=90)
        os.remove(spect_path)

    pdf_file = 'heart_disease_report.pdf'
    pdf.output(pdf_file)
    return pdf_file

# Streamlit UI
st.title("Heart Disease Prediction App")

name = st.text_input("Name")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, step=1)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, step=1)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST'])
max_hr = st.number_input("Max HR", min_value=60, max_value=220, step=1)
exercise_angina = st.selectbox("Exercise Angina", ['N', 'Y'])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat'])

ec_image = st.file_uploader("Upload ECG Image", type=['png', 'jpg', 'jpeg'])
spect_image = st.file_uploader("Upload SPECT Image", type=['png', 'jpg', 'jpeg'])

if ec_image:
    st.image(ec_image, caption='ECG Image', use_column_width=True)
if spect_image:
    st.image(spect_image, caption='SPECT Image', use_column_width=True)

input_data = {
    'Age': age, 'Sex': sex, 'ChestPainType': chest_pain, 'RestingBP': resting_bp,
    'Cholesterol': cholesterol, 'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
    'MaxHR': max_hr, 'ExerciseAngina': exercise_angina, 'Oldpeak': oldpeak, 'ST_Slope': st_slope
}

if st.button("Predict Heart Disease"):
    processed_input = preprocess_input(input_data)
    heart_pred = heart_model.predict(processed_input)[0, 0]

    ecg_pred = 0
    if ec_image is not None:
        img = Image.open(ec_image).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        ecg_pred = ec_model.predict(img_array)[0, 0]

    spect_pred = 0
    if spect_image is not None:
        img = Image.open(spect_image).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        spect_pred = spect_model.predict(img_array)[0, 0]

    st.write(f"Heart Disease Prediction: {'Detected' if heart_pred > 0.5 else 'Not Detected'}")
    st.write(f"ECG Prediction: {'Abnormal' if ecg_pred > 0.5 else 'Normal'}")
    st.write(f"SPECT Prediction: {'Abnormal' if spect_pred > 0.5 else 'Normal'}")

    pdf_file = generate_pdf_report(name, age, sex, heart_pred, ecg_pred, spect_pred, Image.open(ec_image), Image.open(spect_image))
    with open(pdf_file, "rb") as file:
        st.download_button(label="Download Report", data=file, file_name=pdf_file, mime="application/pdf")
