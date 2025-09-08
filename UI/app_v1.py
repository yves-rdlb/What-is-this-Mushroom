# app.py
# Streamlit UI for Mushroom Species → Edibility (with safety/abstain gate)

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"          # put your .h5 / .pkl here
LOOKUPS_DIR = PROJECT_ROOT / "lookups"        # put species_edibility.csv here
DEFAULT_INPUT_SIZE = (224, 224)               # adjust per your CNN

# ======== PAGE SETUP ========
st.set_page_config(
    page_title="What Is This Mushroom?",
    page_icon="🍄",
    layout="wide",
)

st.title("🍄 What Is This Mushroom?")
st.caption("Two-stage pipeline: Species → Deterministic edibility, with a safety gate (abstain below threshold).")

# ======== CACHED LOADERS ========
@st.cache_data
def load_edibility_map(csv_path: Path) -> pd.DataFrame:
    """
    Expects columns: species (str), edible (int: 0=edible, 1=not edible) OR edible_label ('Edible'/'Not Edible')
    """
    df = pd.read_csv(csv_path)
    if "edible" not in df.columns and "edible_label" in df.columns:
        df["edible"] = df["edible_label"].map({"Edible": 0, "Not Edible": 1}).astype(int)
    # Normalize species naming
    df["species"] = df["species"].str.strip().str.lower()
    return df[["species", "edible"]]

@st.cache_resource
def load_model_tf(model_path: Path):
    """
    Lazy-import TF to avoid slow cold start if not needed.
    Replace with your actual loader.
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_model_sklearn(model_path: Path):
    """Example for a baseline sklearn model (e.g., on handcrafted features)."""
    import joblib
    return joblib.load(model_path)

# ======== PRE/POST PROCESSING ========
def preprocess_image_for_tf(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    """Basic preprocessing; match to your training pipeline."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    # If your model expects normalization/mean-std, do it here.
    return np.expand_dims(arr, axis=0)

def postprocess_softmax_to_series(probs: np.ndarray, class_names: List[str]) -> pd.Series:
    """Turn softmax into a sorted pandas Series."""
    probs = probs.squeeze()
    s = pd.Series(probs, index=class_names).sort_values(ascending=False)
    return s

# ======== INFERENCE WRAPPERS (stub these with your real logic) ========
def list_available_models() -> Dict[str, Dict]:
    """
    Register your models here.
    - key: display name in dropdown
    - value: dict with type ('tf'/'sklearn'), path, input_size, classes json/csv
    """
    return {
        "TF_CNN_v1 (224)": {
            "type": "tf",
            "path": MODELS_DIR / "tf_cnn_v1.h5",
            "input_size": (224, 224),
            "classes": LOOKUPS_DIR / "species_classes_v1.json",  # list[str]
        },
        "TF_CNN_v2 (299)": {
            "type": "tf",
            "path": MODELS_DIR / "tf_cnn_v2.h5",
            "input_size": (299, 299),
            "classes": LOOKUPS_DIR / "species_classes_v2.json",
        },
        # Example extra:
        # "Sklearn_Baseline": {
        #     "type": "sklearn",
        #     "path": MODELS_DIR / "baseline.pkl",
        #     "input_size": DEFAULT_INPUT_SIZE,
        #     "classes": LOOKUPS_DIR / "species_classes_baseline.json",
        # },
    }

def load_class_names(path: Path) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)

def predict_species(
    model_meta: Dict,
    img: Image.Image
) -> Tuple[pd.Series, str]:
    """
    Returns:
      - pd.Series of class probabilities (descending)
      - warning string ('' if ok)
    """
    classes = load_class_names(model_meta["classes"])
    input_size = tuple(model_meta.get("input_size", DEFAULT_INPUT_SIZE))
    mtype = model_meta["type"]
    warning = ""

    if mtype == "tf":
        try:
            model = load_model_tf(model_meta["path"])
        except Exception as e:
            return pd.Series(dtype=float), f"Model load error: {e}"
        x = preprocess_image_for_tf(img, input_size)
        # If your TF model outputs logits, apply softmax; if already softmax, skip.
        preds = model.predict(x, verbose=0)
        # Ensure shape (1, num_classes)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds
        else:
            warning = "Unexpected prediction shape; check model output."
            probs = np.array([np.zeros(len(classes))])
        s = postprocess_softmax_to_series(probs, classes)
        return s, warning

    elif mtype == "sklearn":
        # Placeholder: you’d extract features from the image then call predict_proba
        try:
            model = load_model_sklearn(model_meta["path"])
        except Exception as e:
            return pd.Series(dtype=float), f"Model load error: {e}"
        # TODO: implement feature extraction pipeline used for training
        warning = "Sklearn path is stubbed. Implement feature extraction to use sklearn."
        return pd.Series(dtype=float), warning

    else:
        return pd.Series(dtype=float), "Unknown model type."

def lookup_edibility(species: str, edibility_df: pd.DataFrame) -> str:
    row = edibility_df[edibility_df["species"] == species.strip().lower()]
    if row.empty:
        return "Unknown"
    return "Not Edible" if int(row.iloc[0]["edible"]) == 1 else "Edible"

# ======== SIDEBAR CONTROLS ========
with st.sidebar:
    st.header("Settings")
    models = list_available_models()
    model_names = list(models.keys())
    chosen_model_name = st.selectbox("Model", model_names)
    model_meta = models[chosen_model_name]

    conf_threshold = st.slider(
        "Safety threshold (abstain if max prob below)",
        min_value=0.50, max_value=0.99, step=0.01, value=0.80
    )
    top_k = st.slider("Show top-K classes", min_value=3, max_value=10, value=5, step=1)

    st.markdown("---")
    st.caption("Tip: keep the threshold strict for demo safety.")

# ======== MAIN UI ========
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Upload a mushroom image")
    uploaded = st.file_uploader("PNG/JPEG", type=["png", "jpg", "jpeg"])

with col_right:
    st.subheader("2) Run model")
    run = st.button("Predict", use_container_width=True, disabled=(uploaded is None))

# Load edibility table
edibility_csv = LOOKUPS_DIR / "species_edibility.csv"
if not edibility_csv.exists():
    st.warning(f"Missing edibility CSV at: {edibility_csv}. Add it to proceed.")
edibility_df = load_edibility_map(edibility_csv) if edibility_csv.exists() else pd.DataFrame(columns=["species", "edible"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Uploaded image", use_container_width=True)

    if run:
        with st.spinner("Running inference…"):
            probs_series, warn = predict_species(model_meta, img)

        if warn:
            st.warning(warn)

        if probs_series.empty:
            st.error("No predictions returned. Check model wiring.")
        else:
            # Safety/abstain gate
            top_species = probs_series.index[0]
            top_prob = float(probs_series.iloc[0])

            # Tabs for a clean demo
            tab_res, tab_conf, tab_about = st.tabs(["Result", "Confidence", "About"])

            with tab_res:
                if top_prob < conf_threshold:
                    st.error(f"Model abstains (max confidence {top_prob:.2%} < threshold {conf_threshold:.0%}).")
                    st.caption("For safety, we do not make a species or edibility claim below the confidence threshold.")
                else:
                    edibility_label = lookup_edibility(top_species, edibility_df)
                    st.success(f"Predicted species: **{top_species}**  \nEdibility: **{edibility_label}**  \nConfidence: **{top_prob:.2%}**")

                    if edibility_label == "Not Edible":
                        st.warning("⚠️ Not Edible. Never eat wild mushrooms based on an app prediction.")
                    elif edibility_label == "Edible":
                        st.info("Edible label is informational only. Do **not** consume wild mushrooms without expert verification.")

            with tab_conf:
                st.markdown("**Top-K class probabilities**")
                st.bar_chart(probs_series.head(top_k).to_frame("probability"))

                st.write(
                    pd.DataFrame({
                        "species": probs_series.head(top_k).index,
                        "probability": probs_series.head(top_k).values
                    })
                )

            with tab_about:
                st.markdown(
                    """
                    **Pipeline:**
                    1) CNN predicts species from the image
                    2) Deterministic lookup maps species → edibility
                    3) Safety gate abstains if confidence below threshold

                    **Notes:**
                    - This tool is for research/demo only.
                    - Do **not** use for foraging/consumption decisions.
                    - Confidence threshold can be tuned in the sidebar.
                    """
                )
else:
    st.info("Upload an image to begin.")

# ======== FOOTER ========
with st.expander("Debug / paths"):
    st.code(f"Models dir: {MODELS_DIR}\nLookups dir: {LOOKUPS_DIR}\nChosen model: {chosen_model_name}", language="text")
