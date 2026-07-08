from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st

try:
    from config import (
        AZURE_API_KEY,
        AZURE_ENDPOINT_URL,
        DEFAULT_THRESHOLD,
        REQUEST_TIMEOUT_SECONDS,
    )
except ImportError:
    st.error(
        "Nedostaje streamlit_app/config.py. "
        "Kopiraj config.example.py u config.py i upiši Azure endpoint URL i API key."
    )
    st.stop()


def normalize_response(response_json: Any) -> dict[str, Any]:
    if isinstance(response_json, str):
        response_json = json.loads(response_json)

    if isinstance(response_json, list):
        if not response_json:
            raise ValueError("Azure endpoint returned an empty list.")
        return response_json[0]

    if isinstance(response_json, dict):
        if "error" in response_json:
            raise ValueError(response_json["error"])
        return response_json

    raise ValueError(f"Unsupported response format: {type(response_json)}")


def call_azure_endpoint(input_data: dict[str, Any], api_key: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "data": [input_data]
    }

    response = requests.post(
        AZURE_ENDPOINT_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Azure endpoint error {response.status_code}: {response.text}"
        )

    return normalize_response(response.json())


st.set_page_config(
    page_title="Student Depression Prediction",
    page_icon="",
    layout="centered",
)

st.title("Student Depression Prediction")
st.write(
    "Aplikacija šalje podatke na Azure ML Managed Online Endpoint i prikazuje predikciju modela."
)

st.warning(
    "Ovaj model služi isključivo za demonstraciju strojnog učenja i nije medicinski dijagnostički alat."
)

with st.sidebar:
    st.header("Azure pristup")
    entered_key = st.text_input("Azure API key", type="password")

    if not entered_key:
        st.info("Unesi Azure API key iz Azure ML Consume taba.")
        st.stop()

    if entered_key != AZURE_API_KEY:
        st.error("Neispravan API key.")
        st.stop()

st.subheader("Ulazni podaci")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=22, step=1)

    academic_pressure = st.number_input(
        "Academic Pressure", min_value=0, max_value=5, value=3, step=1
    )
    work_pressure = st.number_input(
        "Work Pressure", min_value=0, max_value=5, value=0, step=1
    )
    cgpa = st.number_input(
        "CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1
    )

    study_satisfaction = st.number_input(
        "Study Satisfaction", min_value=0, max_value=5, value=3, step=1
    )
    job_satisfaction = st.number_input(
        "Job Satisfaction", min_value=0, max_value=5, value=0, step=1
    )

    sleep_duration = st.selectbox(
        "Sleep Duration",
        [
            "Less than 5 hours",
            "5-6 hours",
            "7-8 hours",
            "More than 8 hours",
            "Others",
        ],
    )

    dietary_habits = st.selectbox(
        "Dietary Habits",
        ["Healthy", "Moderate", "Unhealthy", "Others"],
    )

    degree = st.selectbox(
        "Degree",
        [
            "B.Tech",
            "M.Tech",
            "BSc",
            "MSc",
            "B.Com",
            "M.Com",
            "BA",
            "MA",
            "BCA",
            "MCA",
            "PhD",
            "Other",
        ],
    )

    suicidal_thoughts = st.selectbox(
        "Have you ever had suicidal thoughts ?",
        ["No", "Yes"],
    )

    work_study_hours = st.number_input(
        "Work/Study Hours", min_value=0, max_value=24, value=6, step=1
    )

    financial_stress = st.number_input(
        "Financial Stress", min_value=0, max_value=5, value=2, step=1
    )

    family_history = st.selectbox(
        "Family History of Mental Illness",
        ["No", "Yes"],
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Gender": gender,
        "Age": int(age),
        "Academic Pressure": int(academic_pressure),
        "Work Pressure": int(work_pressure),
        "CGPA": float(cgpa),
        "Study Satisfaction": int(study_satisfaction),
        "Job Satisfaction": int(job_satisfaction),
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": int(work_study_hours),
        "Financial Stress": int(financial_stress),
        "Family History of Mental Illness": family_history,
    }

    try:
        result = call_azure_endpoint(input_data, entered_key)

        prediction = int(result.get("prediction"))
        probability = result.get("probability")
        label = result.get("label", "")

        st.subheader("Rezultat")

        if prediction == 1:
            st.error("Predikcija: povećan rizik od depresije")
        else:
            st.success("Predikcija: nema indikacije povećanog rizika")

        st.write(f"Label: `{label}`")

        if probability is not None:
            probability = float(probability)
            st.write(f"Probability: `{probability:.4f}`")

            if probability >= DEFAULT_THRESHOLD:
                st.write("Interpretacija: model procjenjuje viši rizik.")
            else:
                st.write("Interpretacija: model procjenjuje niži rizik.")

        with st.expander("Poslani JSON payload"):
            st.json({"data": [input_data]})

        with st.expander("Azure response"):
            st.json(result)

    except Exception as exc:
        st.error(f"Greška pri pozivu Azure endpointa: {exc}")