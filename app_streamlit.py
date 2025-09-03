import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load trained model
MODEL = joblib.load("breast_cancer_model.pkl")  

st.title("🔬 Breast Cancer Detection & Dataset Analysis")

# =======================
# Upload CSV Dataset
# =======================
st.sidebar.header("Upload Dataset (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded successfully!")
    st.write("### Preview of uploaded dataset")
    st.dataframe(df.head())

    # Histogram selection
    st.write("### Histogram of Selected Feature")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_feature = st.selectbox("Select feature for histogram", numeric_cols)
    if selected_feature:
        fig, ax = plt.subplots()
        df[selected_feature].hist(bins=20, ax=ax)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# =======================
# Input Features for Prediction
# =======================
st.sidebar.header("Input Features for Prediction")

# Define all 30 features expected by the model
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.write("### Enter Feature Values for Prediction")
input_data = {}
for feat in features:
    input_data[feat] = st.number_input(feat, min_value=0.0, value=1.0)

X = pd.DataFrame([input_data])
expected_features = MODEL.n_features_in_

if X.shape[1] > expected_features:
    X = X.iloc[:, :expected_features]
elif X.shape[1] < expected_features:
    st.error(f"❌ Model expects {expected_features} features, but got {X.shape[1]}")
    st.stop()

# =======================
# Prediction & PDF Report
# =======================
if st.button("Predict"):
    prediction = MODEL.predict(X)[0]
    result_text = "Benign" if prediction == 0 else "Malignant"
    st.write(f"### Prediction Result: {result_text}")

    # Generate PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Breast Cancer Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Prediction Result: {result_text}", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt="Input Feature Values:", ln=True)
    for key, value in input_data.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)

    # ✅ Fix: convert bytearray to bytes (no .encode())
    pdf_bytes = bytes(pdf.output(dest="S"))

    # Download button
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="breast_cancer_report.pdf",
        mime="application/pdf"
    )