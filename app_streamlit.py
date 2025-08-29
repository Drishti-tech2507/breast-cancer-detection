import streamlit as st

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa, #e3f2fd, #e8f5e9);
        font-family: 'Poppins', sans-serif;
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #3949ab;
        padding: 12px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .navbar a {
        text-decoration: none;
        margin: 0 12px;
        color: white;
        font-weight: 500;
    }
    .navbar a:hover { color: #ffeb3b; }

    /* Card */
    .card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
        animation: slideUp 1s ease;
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- NAVBAR ----------------------
st.markdown("""
<div class="navbar">
    <div><b>🩺 CancerDetect</b></div>
    <div>
        <a href="#home">Home</a>
        <a href="#predict">Prediction</a>
        <a href="#about">About</a>
    </div>
    <div>
        <a href="#profile">Profile</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- CONTENT ----------------------
st.markdown('<div class="card" id="home"> <h2>Welcome to Breast Cancer Detection</h2> <p>This app uses ML to predict whether a tumor is malignant or benign.</p> </div>', unsafe_allow_html=True)

st.markdown('<div class="card" id="predict"> <h2>🔮 Prediction Section</h2> <p>Here we will add ML inputs...</p> </div>', unsafe_allow_html=True)

st.markdown('<div class="card" id="about"> <h2>ℹ️ About</h2> <p>Built with Streamlit + Scikit-learn</p> </div>', unsafe_allow_html=True)
