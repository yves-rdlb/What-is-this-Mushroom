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
import pydeck as pdk

# ---------- PATHS ----------
# Resolve project root as the repo root (one level up from UI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "model"
LOOKUPS_DIR = PROJECT_ROOT / "lookups"
DISTRIBUTIONS_DIR = PROJECT_ROOT / "distributions"
POINTS_CSV = DISTRIBUTIONS_DIR / "species_points.csv"
ICON_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/icon/marker.png"
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

@st.cache_data
def load_points(mtime: float | None = None) -> pd.DataFrame:
    """Load species point data for heatmap. Returns empty df if file missing."""
    if not POINTS_CSV.exists():
        return pd.DataFrame(columns=["species", "lat", "lon", "month"])  # graceful empty
    df = pd.read_csv(POINTS_CSV)
    df = df.dropna(subset=["lat", "lon"])  # ensure coordinates present
    # normalize species key to match norm_species()
    df["species"] = (
        df["species"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    )
    # coerce month to int if present
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(0).astype(int)
    return df

def build_heatmap_deck(points_df: pd.DataFrame, species_key: str, month: int | None = None, add_icon: bool = True) -> pdk.Deck:
    """Return a pydeck Deck object for the given normalized species name."""
    sdf = points_df[points_df["species"] == species_key]
    if month is not None and "month" in sdf.columns:
        sdf = sdf[(sdf["month"] == 0) | (sdf["month"] == month)]

    if sdf.empty:
        # UK-ish fallback center
        center_lat, center_lon = 54.0, -2.5
    else:
        center_lat = float(sdf["lat"].mean())
        center_lon = float(sdf["lon"].mean())

    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=sdf,
        get_position='[lon, lat]',
        aggregation="MEAN",
        radiusPixels=60,
        intensity=2.0,
    )

    # Red dots so sparse datasets are always visible
    dot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=sdf.assign(name=species_key),
        get_position='[lon, lat]',
        get_radius=200,
        radius_min_pixels=4,
        radius_max_pixels=12,
        get_fill_color=[255, 0, 0, 180],
        pickable=True,
    )

    layers = [heat_layer, dot_layer]

    # Optional centroid icon (mean of points) to indicate typical region
    if add_icon and not sdf.empty:
        icon_df = pd.DataFrame([
            {
                "lat": center_lat,
                "lon": center_lon,
                "name": species_key,
                "icon_data": {
                    "url": ICON_URL,
                    "width": 128,
                    "height": 128,
                    "anchorY": 128,
                },
            }
        ])
        icon_layer = pdk.Layer(
            "IconLayer",
            data=icon_df,
            get_icon="icon_data",
            get_position='[lon, lat]',
            get_size=4,
            size_scale=10,
            pickable=True,
        )
        layers.append(icon_layer)

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5, bearing=0, pitch=0)
    return pdk.Deck(
        map_provider='carto',
        map_style='light',
        initial_view_state=view,
        layers=layers,
        tooltip={"text": "{name}"},
    )

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

    with st.expander("API (optional)", expanded=False):
        use_api_a = st.toggle("Use API A", value=False)
        api_a_url = st.text_input("API A URL", "http://127.0.0.1:8000/predict", disabled=not use_api_a)

        use_api_b = st.toggle("Use API B", value=False)
        api_b_url = st.text_input("API B URL", "http://127.0.0.1:8001/predict", disabled=not use_api_b)

        st.caption("If both are ON, the app will call both and compare the results.")
        compare_mode = st.checkbox("Show comparison table", value=True, disabled=not (use_api_a and use_api_b))
        consensus_gate = st.checkbox("Enable consensus gate (require same species & above threshold)", value=False, disabled=not (use_api_a and use_api_b))

    with st.expander("Safety & Display", expanded=True):
        conf_threshold = st.slider("Safety threshold (abstain below)", 0.50, 0.99, 0.85, 0.01)
        st.caption("Lower = more confident predictions shown; higher = stricter abstention.")
        top_k = st.slider("Show top-K", 3, 10, 5, 1)

    with st.expander("Map (optional)"):
        enable_map = st.toggle("Show heatmap", value=True)
        filter_by_month = st.checkbox("Filter by month", value=False)
        selected_month = st.slider(
            "Month", 1, 12, value=pd.Timestamp.now().month, disabled=not filter_by_month
        )
        show_region_pin = st.toggle("Show region pin", value=True)

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

            def _one_api_prediction(url: str) -> Dict:
                r = call_api(url, st.session_state["img_bytes"])
                sp = str(r.get("species", "unknown"))
                cf = float(r.get("confidence", 0.0))
                return {"species": sp, "confidence": cf}

            api_a_res = api_b_res = None

            if use_api_a or use_api_b:
                if use_api_a:
                    api_a_res = _one_api_prediction(api_a_url)
                if use_api_b:
                    api_b_res = _one_api_prediction(api_b_url)

                if use_api_a and use_api_b:
                    # both present → compare and optionally enforce consensus
                    a_sp, a_cf = api_a_res["species"], api_a_res["confidence"]
                    b_sp, b_cf = api_b_res["species"], api_b_res["confidence"]

                    # default pick = higher confidence
                    if a_cf >= b_cf:
                        pick_sp, pick_cf, pick_src = a_sp, a_cf, "API A"
                    else:
                        pick_sp, pick_cf, pick_src = b_sp, b_cf, "API B"

                    if consensus_gate:
                        same_species = (norm_species(a_sp) == norm_species(b_sp))
                        both_above = (a_cf >= conf_threshold and b_cf >= conf_threshold)
                        if same_species and both_above:
                            pick_sp = a_sp  # same species
                            pick_cf = (a_cf + b_cf) / 2.0
                            pick_src = "Consensus (A+B)"
                        else:
                            pick_sp, pick_cf, pick_src = "abstain", 0.0, "Consensus failed"

                    top_species, top_prob = pick_sp, float(pick_cf)
                    # For Confidence tab: store both
                    probs_series = pd.Series({
                        f"API A: {a_sp}": a_cf if api_a_res else 0.0,
                        f"API B: {b_sp}": b_cf if api_b_res else 0.0,
                    }).sort_values(ascending=False)

                    st.session_state["_api_compare"] = {
                        "A": api_a_res,
                        "B": api_b_res,
                        "pick_src": pick_src,
                        "consensus_gate": bool(consensus_gate),
                        "compare_mode": bool(compare_mode),
                    }
                else:
                    # only one API enabled
                    single = api_a_res if use_api_a else api_b_res
                    top_species = single["species"]
                    top_prob = float(single["confidence"])
                    probs_series = pd.Series({top_species: top_prob}).sort_values(ascending=False)
                    st.session_state["_api_compare"] = None
            else:
                # local inference
                probs_series = local_predict(model_meta, st.session_state["img_pil"])
                top_species = probs_series.index[0]
                top_prob = float(probs_series.iloc[0])
                st.session_state["_api_compare"] = None

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            probs_series = pd.Series(dtype=float)

    if not probs_series.empty:
        tab_res, tab_conf, tab_about, tab_map = st.tabs(["Result", "Confidence", "About", "Map"])

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

                api_cmp = st.session_state.get("_api_compare")
                if api_cmp:
                    status = "Consensus required" if api_cmp["consensus_gate"] else "Comparison only"
                    st.caption(f"API mode: {status} • Picked: {api_cmp['pick_src']}")

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
            api_cmp = st.session_state.get("_api_compare")
            if api_cmp and compare_mode:
                a = api_cmp["A"]; b = api_cmp["B"]
                cmp_df = pd.DataFrame([
                    {"source": "API A", "species": a["species"], "confidence": a["confidence"]} if a else {"source": "API A", "species": "—", "confidence": 0.0},
                    {"source": "API B", "species": b["species"], "confidence": b["confidence"]} if b else {"source": "API B", "species": "—", "confidence": 0.0},
                ])
                st.markdown("**API comparison**")
                st.dataframe(cmp_df, use_container_width=True)

        with tab_about:
            st.markdown(
                """
                **Pattern from lesson:** Frontend (Streamlit) → Requests → API → JSON → Display
                **Fallback:** Local TensorFlow inference for offline demo
                **Pipeline:** *Species → Edibility (deterministic)* with a **safety threshold** (abstain)
                """
            )
        with tab_map:
            # guardrails: only show when above threshold and user enabled
            if top_prob < conf_threshold:
                st.info("Abstained below safety threshold — map hidden.")
            elif 'enable_map' in locals() and not enable_map:
                st.caption("Heatmap disabled in sidebar.")
            else:
                # include file mtime so cache invalidates when CSV changes
                mtime = POINTS_CSV.stat().st_mtime if POINTS_CSV.exists() else None
                pts_df = load_points(mtime)
                # If no data file or no matching rows, inform the user explicitly
                if pts_df.empty:
                    st.warning("No distribution data loaded. Add rows to distributions/species_points.csv to see the heatmap.")
                species_key = norm_species(top_species)
                sdf = pts_df[pts_df["species"] == species_key]
                with st.expander("Debug (map data)"):
                    st.write(f"CSV path: {POINTS_CSV}")
                    st.write({
                        "total_rows": int(pts_df.shape[0]),
                        "species_rows": int(sdf.shape[0]),
                        "unique_species_keys": int(pts_df["species"].nunique()) if not pts_df.empty else 0,
                    })
                    st.dataframe(sdf.head(20), use_container_width=True)
                if not pts_df.empty and sdf.empty:
                    st.info(f"No points found for species '{species_key}'. Add rows in species_points.csv (species, lat, lon, month).")
                month_arg = selected_month if ('filter_by_month' in locals() and filter_by_month) else None
                deck = build_heatmap_deck(pts_df, species_key, month=month_arg, add_icon=('show_region_pin' in locals() and show_region_pin))
                st.pydeck_chart(deck)
                st.caption("Distribution is illustrative. Do not forage based solely on this app.")
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
