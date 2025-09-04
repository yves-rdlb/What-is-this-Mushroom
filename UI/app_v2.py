
# app.py
# Streamlit UI for Mushroom Species → Edibility with API-first pattern (lesson style)

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOOKUPS_DIR = PROJECT_ROOT / "lookups"
DEFAULT_INPUT_SIZE = (224, 224)

# ---------- PAGE ----------
st.set_page_config(page_title="What Is This Mushroom?", page_icon="🍄", layout="wide")
st.title("🍄 What Is This Mushroom?")
st.caption("Lesson-style API-first UI. Falls back to local inference if API is not set.")

# ---------- HELPERS ----------
@st.cache_data
def load_edibility_map(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expect columns: species, edible (0=edible,1=not)
    df["species"] = (
        df["species"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df[["species", "edible"]]

def norm_species(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

@st.cache_resource
def load_model_tf(model_path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

def preprocess_image_for_tf(img: Image.Image, size: Tuple[int, int], scheme: str) -> np.ndarray:
    # Match to your training preproc; examples for common backbones:
    arr = np.asarray(img.convert("RGB").resize(size)).astype("float32")
    if scheme == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    elif scheme == "xception":
        from tensorflow.keras.applications.xception import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, 0)

def to_probs(preds: np.ndarray) -> np.ndarray:
    arr = preds.squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")
    # If not normalized, softmax it
    if not np.isclose(arr.sum(), 1.0, atol=1e-3):
        e = np.exp(arr - arr.max())
        arr = e / e.sum()
    return arr

def list_available_models() -> Dict[str, Dict]:
    return {
        "CNN_v1 (224)": {
            "type": "tf",
            "path": MODELS_DIR / "cnn_v1.keras",
            "input_size": (224, 224),
            "classes": LOOKUPS_DIR / "species_classes_v1.json",
            "preproc": "efficientnet",  # change to your actual scheme
        },
        "CNN_v2 (299)": {
            "type": "tf",
            "path": MODELS_DIR / "cnn_v2.keras",
            "input_size": (299, 299),
            "classes": LOOKUPS_DIR / "species_classes_v2.json",
            "preproc": "xception",  # change to your actual scheme
        },
    }

def load_class_names(path: Path) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)

def local_predict(model_meta: Dict, img: Image.Image) -> pd.Series:
    classes = load_class_names(model_meta["classes"])
    model = load_model_tf(model_meta["path"])
    x = preprocess_image_for_tf(img, tuple(model_meta["input_size"]), model_meta.get("preproc", "custom"))
    preds = model.predict(x, verbose=0)
    probs = to_probs(preds)
    s = pd.Series(probs, index=classes).sort_values(ascending=False)
    return s

def call_api(api_url: str, image_bytes: bytes) -> Dict:
    import requests
    # lesson style: send a multipart file and get JSON back
    files = {"file": ("upload.jpg", image_bytes, "image/jpeg")}
    r = requests.post(api_url, files=files, timeout=30)
    r.raise_for_status()
    return r.json()  # expected: {"species": str, "confidence": float}

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Settings")
    models = list_available_models()
    model_name = st.selectbox("Local model", list(models.keys()))
    model_meta = models[model_name]

    use_api = st.toggle("Use team API (lesson style)", value=False)
    api_url = st.text_input("API URL", "http://127.0.0.1:8000/predict", disabled=not use_api)

    conf_threshold = st.slider("Safety threshold (abstain below)", 0.50, 0.99, 0.85, 0.01)
    top_k = st.slider("Show top-K", 3, 10, 5, 1)

# ---------- DATA ----------
edibility_csv = LOOKUPS_DIR / "species_edibility.csv"
if not edibility_csv.exists():
    st.warning(f"Missing edibility CSV at: {edibility_csv}")
edibility_df = load_edibility_map(edibility_csv) if edibility_csv.exists() else pd.DataFrame(columns=["species","edible"])

# ---------- UI ----------
colL, colR = st.columns([1, 1])
with colL:
    st.subheader("1) Upload")
    uploaded = st.file_uploader("PNG/JPEG", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(io.BytesIO(uploaded.read()))
        st.image(img, caption="Uploaded image", use_container_width=True)
with colR:
    st.subheader("2) Predict")
    run = st.button("Go", use_container_width=True, disabled=(uploaded is None))

if uploaded and run:
    with st.spinner("Predicting…"):
        try:
            if use_api:
                # LESSON: frontend → requests → API → JSON
                result = call_api(api_url, uploaded.getvalue())
                top_species = result.get("species", "unknown")
                top_prob = float(result.get("confidence", 0.0))
                probs_series = pd.Series({top_species: top_prob}).sort_values(ascending=False)
            else:
                # fallback: local inference for offline demo
                probs_series = local_predict(model_meta, img)
                top_species = probs_series.index[0]
                top_prob = float(probs_series.iloc[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            probs_series = pd.Series(dtype=float)

    if not probs_series.empty:
        tab_res, tab_conf, tab_about = st.tabs(["Result", "Confidence", "About"])

        with tab_res:
            if top_prob < conf_threshold:
                st.error(f"Model abstains (max {top_prob:.2%} < threshold {conf_threshold:.0%}).")
                st.caption("Following the safety gate rule to avoid risky advice.")
            else:
                pred_norm = norm_species(top_species)
                row = edibility_df[edibility_df["species"] == pred_norm]
                if row.empty:
                    edibility = "Unknown"
                else:
                    edibility = "Not Edible" if int(row.iloc[0]["edible"]) == 1 else "Edible"

                st.success(f"Species: **{top_species}**")
                st.info(f"Edibility: **{edibility}**")
                st.write(f"Confidence: **{top_prob:.2%}**")

                if edibility == "Not Edible":
                    st.warning("⚠️ Do not consume wild mushrooms.")
                elif edibility == "Edible":
                    st.caption("Informational only. Never eat wild mushrooms based on an app prediction.")

        with tab_conf:
            st.markdown("**Top-K probabilities**")
            st.bar_chart(probs_series.head(top_k).to_frame("probability"))
            st.dataframe(
                pd.DataFrame({"species": probs_series.head(top_k).index,
                              "probability": probs_series.head(top_k).values})
            )

        with tab_about:
            st.markdown(
                """
                **Pattern from lesson:** Frontend (Streamlit) → Requests → API → JSON → Display.
                **Fallback:** Local TF inference for offline demo.
                **Pipeline:** Species → Edibility (deterministic) with a safety threshold (abstain).
                """
            )
else:
    st.info("Upload an image to begin.")
