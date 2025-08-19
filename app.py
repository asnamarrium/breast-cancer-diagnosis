import streamlit as st
import numpy as np
import joblib

# Load trained models
lr_noFS = joblib.load("logistic_regression_noFS.pkl")
rf_noFS = joblib.load("random_forest_noFS.pkl")
lr_FS = joblib.load("logistic_regression_FS.pkl")
rf_FS = joblib.load("random_forest_FS.pkl")

st.title("🔬 Breast Cancer Prediction App")

st.sidebar.header("Select Model")
model_option = st.sidebar.selectbox(
    "Choose a Model",
    (
        "Logistic Regression (No Feature Selection)",
        "Random Forest (No Feature Selection)",
        "Logistic Regression (With Feature Selection)",
        "Random Forest (With Feature Selection)"
    )
)

# Take user inputs (based on 30 features of dataset)
st.header("Enter Patient Features")
features = []

for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

features = np.array(features).reshape(1, -1)

# Choose model
if model_option == "Logistic Regression (No Feature Selection)":
    model = lr_noFS
elif model_option == "Random Forest (No Feature Selection)":
    model = rf_noFS
elif model_option == "Logistic Regression (With Feature Selection)":
    model = lr_FS
else:
    model = rf_FS

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer)")

    st.subheader("Probability Scores")
    st.write(f"Benign: {prob[0]*100:.2f}%")
    st.write(f"Malignant: {prob[1]*100:.2f}%")
