"""Streamlit app for heart disease prediction."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path(__file__).parent
MODEL_PATH = ARTIFACT_DIR / "heart_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.pkl"
LABEL_ENCODERS_PATH = ARTIFACT_DIR / "label_encoders.pkl"

FEATURE_INFO = {
    "Age": ("Age (years)", "number", 28, 77, 54),
    "Sex": ("Sex", "select", ["F", "M"], "M"),
    "ChestPainType": (
        "Chest pain type",
        "select",
        ["ATA", "NAP", "ASY", "TA"],
        "ATA",
    ),
    "RestingBP": ("Resting blood pressure (mm Hg)", "number", 0, 200, 130),
    "Cholesterol": ("Serum cholesterol (mg/dl)", "number", 0, 603, 246),
    "FastingBS": ("Fasting blood sugar > 120 mg/dl", "select", [0, 1], 0),
    "RestingECG": (
        "Resting ECG results",
        "select",
        ["Normal", "ST", "LVH"],
        "Normal",
    ),
    "MaxHR": ("Maximum heart rate achieved", "number", 60, 202, 150),
    "ExerciseAngina": ("Exercise induced angina", "select", ["N", "Y"], "N"),
    "Oldpeak": ("ST depression induced by exercise", "number", -2.6, 6.2, 1.0),
    "ST_Slope": ("ST slope", "select", ["Up", "Flat", "Down"], "Up"),
}


def _artifacts_exist():
    return (
        MODEL_PATH.exists()
        and SCALER_PATH.exists()
        and FEATURE_NAMES_PATH.exists()
        and LABEL_ENCODERS_PATH.exists()
    )


@st.cache_resource
def load_artifacts():
    if not _artifacts_exist():
        from train_model import build_and_train

        build_and_train()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    return model, scaler, feature_names, label_encoders


def encode_features(raw_inputs, feature_names, label_encoders):
    row = {}
    for name in feature_names:
        value = raw_inputs[name]
        if name in label_encoders:
            row[name] = label_encoders[name].transform([value])[0]
        else:
            row[name] = value
    return pd.DataFrame([row], columns=feature_names)


def main():
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon="❤️",
        layout="centered",
    )

    st.title("❤️ Heart Disease Predictor")
    st.markdown(
        "Predict heart disease risk from clinical features using a "
        "K-Nearest Neighbors classifier trained on `heart.csv`."
    )

    model, scaler, feature_names, label_encoders = load_artifacts()
    st.caption(
        f"Model: **KNN** (k={model.n_neighbors}, tuned via 5-fold cross-validation)"
    )

    st.subheader("Patient features")
    inputs = {}

    col1, col2 = st.columns(2)
    for i, name in enumerate(feature_names):
        label, field_type, *rest = FEATURE_INFO[name]
        column = col1 if i % 2 == 0 else col2
        with column:
            if field_type == "select":
                options, default = rest
                default_index = options.index(default) if default in options else 0
                inputs[name] = st.selectbox(label, options, index=default_index, key=name)
            else:
                min_val, max_val, default = rest
                step = 0.1 if name == "Oldpeak" else 1.0
                inputs[name] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default),
                    step=step,
                    key=name,
                )

    if st.button("Predict", type="primary", use_container_width=True):
        features = encode_features(inputs, feature_names, label_encoders)
        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])
        probabilities = model.predict_proba(scaled)[0]
        risk_pct = probabilities[1] * 100

        if prediction == 1:
            st.error(f"**Prediction: Heart disease detected** ({risk_pct:.1f}% probability)")
        else:
            st.success(f"**Prediction: No heart disease** ({100 - risk_pct:.1f}% probability)")
        st.caption(f"Disease probability: {risk_pct:.1f}% | No disease: {100 - risk_pct:.1f}%")

    with st.expander("About this model"):
        st.markdown(
            """
            - **Dataset:** Heart Failure Prediction Dataset (`heart.csv`, 918 rows)
            - **Preprocessing:** Label encoding for categorical features, StandardScaler on all features
            - **Model:** KNN with `n_neighbors` tuned from 1–20 via GridSearchCV (5-fold CV)
            - **Split:** 80% train / 20% test, random_state=42
            """
        )


if __name__ == "__main__":
    main()
