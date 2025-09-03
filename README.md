🩺 Breast Cancer Detection & Analysis

This project uses Machine Learning (ML) to analyze and predict breast cancer diagnoses based on input features. It includes a trained model, API endpoints, and an interactive Streamlit dashboard for visualization and prediction.

📂 Project Structure
├── .venv/                 # Virtual environment  
├── .vscode/               # VS Code settings  
├── model/                 # ML model scripts  
│   ├── app.py             # Flask/ML app  
│   ├── breast_cancer_model.py  # Model definition/training logic  
│   ├── predict_tumor.py   # Tumor prediction logic  
│   ├── predict.py         # Prediction handler  
│   ├── train.py           # Training script  
│   ├── utils.py           # Utility functions  
│   ├── saved_model.pkl    # Trained model file  
├── static/                # Static files (CSS, images)  
│   ├── style.css  
├── templates/             # HTML templates (if Flask is used)  
├── app_streamlit.py       # Streamlit UI for predictions  
├── breast_cancer_model.ipynb # Jupyter notebook for experiments  
├── requirements.txt       # Dependencies  
├── save_model.py          # Script to save trained model  
├── saved_model.pkl        # Final trained model  

 Installation
	1.	Clone the repository:
 git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
	2.	Create and activate a virtual environment:
 python -m venv .venv
source .venv/bin/activate   # Mac/Linux  
.venv\Scripts\activate      # Windows  
3.	Install dependencies:
pip install -r requirements.txt

Usage

1. Train Model

If you want to retrain the model:
python model/train.py

This will generate/update saved_model.pkl.

2. Run Streamlit App
streamlit run app_streamlit.py

📊 Features
	•	Data preprocessing & feature scaling
	•	Breast cancer classification (Benign / Malignant)
	•	Interactive web UI with Streamlit
	•	Option to upload patient data for prediction
	•	Model saved as .pkl for easy reuse
🧠 Model

The model is trained using the Breast Cancer Wisconsin (Diagnostic) Dataset.
	•	Algorithm: Logistic Regression / Random Forest / (your chosen ML model)
	•	Evaluation metrics: Accuracy, Precision, Recall, F1-score

 Requirements

Main libraries:
	•	scikit-learn
	•	pandas
	•	numpy
	•	matplotlib
	•	streamlit
👩‍⚕️ Use Case

This project can assist in early detection of breast cancer by providing a quick, data-driven prediction. It is not a substitute for medical diagnosis and should be used for educational/research purposes only.
