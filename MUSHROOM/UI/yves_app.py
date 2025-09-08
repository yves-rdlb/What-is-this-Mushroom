url="http://127.0.0.1:8000/predict/"

import streamlit as st
import requests

st.title("🍄 What is this Mushroom?")

uploaded_file = st.file_uploader("Upload a mushroom image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # envoie au backend FastAPI
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
    res = requests.post("http://127.0.0.1:8000/predict/", files=files)

    if res.status_code == 200:
        data = res.json()["prediction"]
        st.success(f"✅ Prediction: {data['class']} ({data['edibility']})")
        st.write(f"Confidence: {data['confidence']}")
    else:
        st.error(f"Error: {res.text}")
