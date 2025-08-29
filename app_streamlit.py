import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

# =========================
# Custom CSS + Animation
# =========================
page_style = """
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #f8f9fa, #e3f2fd, #e8f5e9);
    font-family: 'Poppins', sans-serif;
}

/* Header Bar */
.header {
    background: #3949ab;
    color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: 600;
    animation: fadeIn 2s ease-in-out;
}

/* Navigation Bar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #5c6bc0;
    padding: 12px 30px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.navbar-left { font-size: 22px; font-weight: 600; color: white; }
.navbar-center a, .navbar-right a {
    margin: 0 12px;
    text-decoration: none;
    color: white;
    font-weight: 500;
    transition: 0.3s;
}
.navbar-center a:hover, .navbar-right a:hover {
    color: #ffeb3b;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin: 20px 0;
    animation: slideUp 1s ease;
}

/* Animations */
@keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
@keyframes slideUp { from {transform: translateY(30px); opacity:0;} to {transform: translateY(0); opacity:1;} }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# =========================
# Navigation Header
# =========================
st.markdown("""
<div class="navbar">
    <div class="navbar-left">🩺 CancerDetect</div>
    <div class="navbar-center">
        <a href="#home">Home</a>
        <a href="#predict">Prediction</a>
        <a href="#about">About</a>
    </div>
    <div class="navbar-right">
        <a href="#profile">Profile</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# ML Model Setup
# =========================
@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, data

model, data = train_model()

# =========================
# Home Section
# =========================
st.markdown('<div class="header" id="home">Welcome to Breast Cancer Detection</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("This app uses **Machine Learning** to predict whether a tumor is **Malignant** or **Benign** based on input features.")
    st.write("Navigate to **Prediction** to test with real data.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prediction Section
# =========================
st.markdown('<div class="header" id="predict">🧪 Prediction</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Enter tumor features to get prediction:")

    feature_input = []
    cols = st.columns(3)
    for i, feature in enumerate(data.feature_names):
        with cols[i % 3]:
            value = st.number_input(f"{feature}", float(data.data[:, i].min()), float(data.data[:, i].max()), float(np.median(data.data[:, i])))
            feature_input.append(value)

    if st.button("🔍 Predict"):
        prediction = model.predict([feature_input])[0]
        result = "Malignant ❌ (Cancerous)" if prediction == 0 else "Benign ✅ (Non-Cancerous)"
        st.success(f"### Result: {result}")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# About Section
# =========================
st.markdown('<div class="header" id="about">ℹ️ About</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("This project is built using **Streamlit**, **Scikit-learn**, and **Python**.")
    st.write("The model is trained on the Breast Cancer Wisconsin dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
