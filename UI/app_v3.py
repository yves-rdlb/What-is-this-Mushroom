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
# Anchor paths at the repo root (one level up from /UI)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "model"
LOOKUPS_DIR = PROJECT_ROOT / "lookups"
DEFAULT_INPUT_SIZE = (224, 224)

# ---------- PAGE ----------
st.set_page_config(page_title="What Is This Mushroom?", page_icon="🍄", layout="wide")

# --- Light design polish (CSS) ---
st.markdown(
    """
    <style>
      /* overall look */
      .main {
        background: radial-gradient(1000px 500px at 20% 0%, #f6fff7 0%, #ffffff 40%) no-repeat;
      }
      h1, h2, h3 { letter-spacing: 0.2px; }
      /* cards */
      .card {
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        background: #fff;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
      }
      /* pill labels */
      .pill {
        display:inline-block; padding:5px 10px; border-radius:999px;
        font-size:12px; font-weight:600; letter-spacing:.2px;
        background:#eef6ff; color:#1250aa; border:1px solid #d9eaff;
      }
      .pill.ok { background:#edfbf0; color:#126b36; border-color:#c9f1d6; }
      .pill.warn { background:#fff4ed; color:#9a3e00; border-color:#ffe1cc; }
      .pill.unknown { background:#f3f4f6; color:#374151; border-color:#e5e7eb; }
      /* CTA button */
      div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 10px 16px;
      }
      /* footer */
      .footer {
        color:#6b7280; font-size:12px; margin-top:8px;
      }
      /* nicer tabs spacing */
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        padding-top: 8px; padding-bottom: 8px;
      }
      /* dataframes tighter */
      .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("### 🍄 What Is This Mushroom?")
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

# put this near the top of HELPERS, above load_model_tf
try:
    # if your training code defines vit_layer here, use the real one
    from loading.load_data import vit_layer as _vit_layer
except Exception:
    _vit_layer = None    # we'll fall back to a safe stub


@st.cache_resource
def load_model_tf(model_path: Path):
    """
    Loads a Keras model. Tries safe deserialization first; if the model
    contains Lambda layers or other Python callables, retries with
    custom_objects and safe_mode=False.

    Only enable unsafe mode for trusted model files.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ---- register any custom callables used in Lambda layers ----
    # If you have more (e.g., gelu_fn, patchify, etc.), add them here.
    custom_objects = {
        "vit_layer": _vit_layer,   # real import if available, else stub above
    }

    # 1) Try normal (safe) load first
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,  # make vit_layer visible
        )
    except Exception as e:
        msg = str(e)

        # 2) If blocked by Lambda / unsafe deserialization, enable unsafe and retry
        if "unsafe_deserialization" in msg or "Lambda layer" in msg:
            try:
                from keras import config as keras_config
                keras_config.enable_unsafe_deserialization()
            except Exception:
                pass

            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False,   # allow Python callable deserialization
            )

        # Otherwise bubble up
        raise



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
        "ViT": {
            "type": "tf",
            "path": MODELS_DIR / "ViT_model.keras",
            "input_size": (224, 224),
            "classes": LOOKUPS_DIR / "species_classes.json",
            "preproc": "custom",  # match your training
        },
        "EffNetB2_finetuned": {
            "type": "tf",
            "path": MODELS_DIR / "mushroom_model_EfficientNetV2B0_finetuned.keras",
            "input_size": (224, 224),
            "classes": LOOKUPS_DIR / "species_classes.json",
            "preproc": "efficientnet",
        },
        "EffNetV2B0_iter6": {
            "type": "tf",
            "path": MODELS_DIR / "mushroom_model_EfficientNetV2B0_6.keras",
            "input_size": (224, 224),
            "classes": LOOKUPS_DIR / "species_classes.json",
            "preproc": "efficientnet",
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
    with st.expander("Model", expanded=True):
        models = list_available_models()
        model_name = st.selectbox("Local model", list(models.keys()))
        model_meta = models[model_name]

    with st.expander("API (optional)"):
        use_api = st.toggle("Use team API", value=False)
        api_url = st.text_input("API URL", "http://127.0.0.1:8000/predict", disabled=not use_api)

    with st.expander("Safety & Display", expanded=True):
        conf_threshold = st.slider("Safety threshold (abstain below)", 0.50, 0.99, 0.85, 0.01)
        st.caption("Lower = more confident predictions shown; higher = stricter abstention.")
        top_k = st.slider("Show top-K", 3, 10, 5, 1)

# ---------- DATA ----------
edibility_csv = LOOKUPS_DIR / "species_edibility.csv"
if not edibility_csv.exists():
    st.warning(f"Missing edibility CSV at: {edibility_csv}")
edibility_df = load_edibility_map(edibility_csv) if edibility_csv.exists() else pd.DataFrame(columns=["species","edible"])

# ---------- UI ----------
left, right = st.columns([1, 1])

with left:
    st.markdown("#### 1) Upload")
    uploaded = st.file_uploader("Drag & drop a PNG/JPEG or browse files", type=["png", "jpg", "jpeg"])

    if uploaded:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        img_bytes = uploaded.getvalue()
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, caption="Preview", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Tip: use a clear, centered photo with good lighting for best results.")


with right:
    st.markdown("#### 2) Predict")
    if uploaded:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        run = st.button("⚡ Go", use_container_width=True, disabled=False)
        placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Upload an image first to enable prediction.")


# ---------- INFERENCE ----------
if uploaded and run:
    with st.spinner("Predicting…"):
        try:
            if st.session_state.get("img_bytes") is None:
                # store once for API path reuse
                st.session_state["img_bytes"] = uploaded.getvalue()
            if st.session_state.get("img_pil") is None:
                st.session_state["img_pil"] = img

            if 'use_api' not in locals():
                use_api = False  # safety

            if use_api:
                result = call_api(api_url, st.session_state["img_bytes"])
                top_species = result.get("species", "unknown")
                top_prob = float(result.get("confidence", 0.0))
                probs_series = pd.Series({top_species: top_prob}).sort_values(ascending=False)
            else:
                probs_series = local_predict(model_meta, st.session_state["img_pil"])
                top_species = probs_series.index[0]
                top_prob = float(probs_series.iloc[0])

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            probs_series = pd.Series(dtype=float)

    if not probs_series.empty:
        tab_res, tab_conf, tab_about = st.tabs(["Result", "Confidence", "About"])

        with tab_res:
            if top_prob < conf_threshold:
                st.markdown('<span class="pill warn">Abstained</span>', unsafe_allow_html=True)
                st.error(f"Max confidence {top_prob:.2%} is below safety threshold {conf_threshold:.0%}.")
                st.caption("Following the safety gate rule to avoid risky advice.")
            else:
                pred_norm = norm_species(top_species)
                row = edibility_df[edibility_df["species"] == pred_norm]
                if row.empty:
                    edibility = "Unknown"
                    pill_cls = "unknown"
                else:
                    edibility = "Not Edible" if int(row.iloc[0]["edible"]) == 1 else "Edible"
                    pill_cls = "warn" if edibility == "Not Edible" else "ok"

                # Summary strip
                c1, c2, c3 = st.columns([1.2, 1, 1])
                with c1:
                    st.subheader(top_species)
                    st.markdown(f'<span class="pill {pill_cls}">Edibility: {edibility}</span>', unsafe_allow_html=True)
                with c2:
                    st.metric("Confidence", f"{top_prob:.1%}")
                with c3:
                    st.metric("Top-K shown", f"{min(top_k, len(probs_series))}")

                if edibility == "Not Edible":
                    st.warning("⚠️ Do not consume wild mushrooms.")
                elif edibility == "Edible":
                    st.caption("Informational only. Never eat wild mushrooms based on an app prediction.")

        with tab_conf:
            st.markdown("**Top-K probabilities**")
            chart_df = probs_series.head(top_k).to_frame("probability")
            st.bar_chart(chart_df)
            st.dataframe(
                pd.DataFrame({
                    "species": chart_df.index,
                    "probability": chart_df["probability"].values
                }),
                use_container_width=True,
            )

        with tab_about:
            st.markdown(
                """
                **Pattern from lesson:** Frontend (Streamlit) → Requests → API → JSON → Display
                **Fallback:** Local TensorFlow inference for offline demo
                **Pipeline:** *Species → Edibility (deterministic)* with a **safety threshold** (abstain)
                """
            )
else:
    st.info("Upload an image to begin.")

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
      Built by Team • v0.2 — Cards, pill statuses, collapsible sidebar, and polished layout added.
    </div>
    """,
    unsafe_allow_html=True,
)
