import streamlit as st
import numpy as np
import joblib

label_map = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}

# Load model dan scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Judul aplikasi
st.title("Prediksi Tingkat Obesitas")

# Input data dari pengguna
age = st.number_input("Usia", min_value=1, max_value=120, value=25)
weight = st.number_input("Berat Badan (kg)", min_value=1.0, value=60.0, step=0.1, format="%.1f")
height = st.number_input("Tinggi Badan (m)", min_value=0.5, max_value=2.5, value=1.7)
gender1 = st.selectbox("Jenis Kelamin", ['Female', 'Male'])
carrier1 = st.selectbox("Riwayat Keluarga Obesitas?", ['yes', 'no'])
calories1 = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", ['yes', 'no'])
vegetables = st.slider("Frekuensi Makan Sayur (1=jarang, 5=sering)", 1, 5, 3)
snack = st.slider("Frekuensi Konsumsi Snack (1=jarang, 5=sering)", 1, 5, 3)
water = st.slider("Konsumsi Air Harian (liter)", 1.0, 5.0, 2.0)

# Encode input kategorikal (pastikan sama seperti saat training)
gender = 1 if gender1 == 'Male' else 0
carrier = 1 if carrier1 == 'yes' else 0
calories = 1 if calories1 == 'yes' else 0

# Gabungkan fitur
input_data = np.array([[age, gender, height, weight, calories,
                          vegetables, water, carrier, snack]])

# Normalisasi
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)[0]
    prediction_label = label_map[prediction]

    st.subheader("Hasil Prediksi:")
    st.success(f"Tingkat Obesitas: {prediction_label}")
