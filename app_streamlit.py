from sklearn.datasets import load_breast_cancer
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# 🎀 Custom Page Config
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎀", layout="wide")

# 🎨 CSS styling
st.markdown("""
    <style>
    .main {background-color: #fff5f7;}
    .stButton>button {
        background-color: #ff4d79;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    h1, h2, h3 {color: #e6005c;}
    </style>
""", unsafe_allow_html=True)

# ------------------ Step 1: Upload or Use Sample Data ------------------
st.title("🎀 Breast Cancer Prediction Dashboard")

uploaded_file = st.file_uploader("📂 Upload your Breast Cancer dataset (CSV)", type="csv")
use_sample = st.button("🧪 Or Use Built-in Breast Cancer Dataset")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Data Preview (Uploaded)")
    st.dataframe(df.head())

elif use_sample:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    st.subheader("🔎 Data Preview (Sample Dataset)")
    st.dataframe(df.head())

# ------------------ Step 2: Continue only if data is loaded ------------------
if df is not None:

    st.subheader("📊 Data Analysis")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    if st.checkbox("Show Heatmap Correlation"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=False, cmap="pink")
        st.pyplot(plt)

    if st.checkbox("Show Distribution Plots"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        col = st.selectbox("Choose feature", numeric_cols)
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, color="hotpink")
        st.pyplot(plt)

    # ------------------ Step 3: Model Training ------------------
    st.subheader("🤖 Model Training")
    target = st.selectbox("Select Target Column", df.columns)
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    # ✅ Fix target column for classification
    if y.dtype == float:  
        y = y.round().astype(int)  # if 0.0 / 1.0 → make it int
    if y.dtype == object:  
        # common breast cancer dataset labels: "M" (malignant), "B" (benign)
        if set(y.unique()) <= {"M", "B"}:
            y = y.map({"M": 0, "B": 1})
        else:
            st.error("⚠️ Target column contains non-numeric labels. Please clean data.")
            st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

    st.write("✅ Model Comparison:", results)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    st.success(f"🏆 Best Model: {best_model_name} with Accuracy {results[best_model_name]:.2f}")

    # ------------------ Step 4: Feature Importance ------------------
    if best_model_name == "Random Forest":
        st.subheader("🌟 Feature Importance")
        importances = best_model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        st.bar_chart(feat_imp)

    # ------------------ Step 5: Predictions ------------------
    st.subheader("🔮 Make a Prediction")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
    
    if st.button("Predict"):
        user_df = pd.DataFrame([input_data])
        prediction = best_model.predict(user_df)[0]
        st.write("🎯 Prediction:", "Benign (1)" if prediction == 1 else "Malignant (0)")

    # ------------------ Step 6: PDF Report ------------------
    def generate_pdf(results, best_model_name):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, 750, "Breast Cancer Report 🎀")
        c.setFont("Helvetica", 12)
        y = 700
        for model, acc in results.items():
            c.drawString(100, y, f"{model}: {acc:.2f}")
            y -= 20
        c.drawString(100, y-20, f"Best Model: {best_model_name}")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    if st.button("📑 Download PDF Report"):
        pdf = generate_pdf(results, best_model_name)
        st.download_button("⬇️ Download Report", data=pdf, file_name="breast_cancer_report.pdf", mime="application/pdf")