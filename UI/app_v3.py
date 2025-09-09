# app.py
# Streamlit UI for Mushroom Species → Edibility with API-first pattern (lesson style)
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
import streamlit as st
import pydeck as pdk
import time
import threading

# ---------- PATHS ----------
# Resolve project root as the repo root (one level up from UI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
st.caption("API-only UI using a single ViT endpoint (no local models).")

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

# (All local model helpers removed; API-only mode)

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

def _parse_confidence_to_float01(val) -> float:
    """Accept 0..1 float, 0..100 float, or '87.2%' string and normalize to 0..1."""
    try:
        if isinstance(val, str):
            v = val.strip()
            if v.endswith("%"):
                return max(0.0, min(1.0, float(v[:-1]) / 100.0))
            # plain string number
            f = float(v)
            # assume 0..100 if > 1
            return f / 100.0 if f > 1.0 else f
        if isinstance(val, (int, float)):
            return float(val) / 100.0 if float(val) > 1.0 else float(val)
    except Exception:
        pass
    return 0.0

def call_api(api_url: str, image_bytes: bytes, timeout_s: float = 30.0) -> Dict:
    """
    Calls the ViT API and normalizes the response into:
        {"species": str, "confidence": float 0..1, "edibility": Optional[str]}
    Supports both the ViT API schema ({"prediction": {"class", "confidence", "edibility"}})
    and a fallback simple schema ({"species", "confidence"}).
    """
    import requests
    files = {"file": ("upload.jpg", image_bytes, "image/jpeg")}
    r = requests.post(api_url, files=files, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    species = None
    conf = None
    edibility = None

    if isinstance(data, dict) and "prediction" in data and isinstance(data["prediction"], dict):
        p = data["prediction"]
        species = p.get("class") or p.get("species") or "unknown"
        conf = _parse_confidence_to_float01(p.get("confidence", 0.0))
        edibility = p.get("edibility")
    else:
        species = data.get("species", "unknown")
        conf = _parse_confidence_to_float01(data.get("confidence", 0.0))
        edibility = data.get("edibility")

    return {"species": str(species), "confidence": float(conf), "edibility": (str(edibility) if edibility is not None else None)}

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Settings")
    with st.expander("ViT API", expanded=True):
        vit_api_url = st.text_input("Endpoint", "http://127.0.0.1:8000/predict/", help="Your ViT FastAPI predict endpoint")
        request_timeout_s = st.number_input(
            "Request timeout (s)", min_value=5, max_value=180, value=60, step=5,
            help="Increase if the API is cold-starting or downloading weights."
        )

    with st.expander("Safety & Display", expanded=True):
        conf_threshold = st.slider("Safety threshold (abstain below)", 0.50, 0.99, 0.85, 0.01)
        st.caption("Lower = more confident predictions shown; higher = stricter abstention.")

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
    st.markdown("#### 1) Image source")
    source = st.radio("Choose input", ["Upload", "Camera"], horizontal=True)

    uploaded = None
    if source == "Upload":
        uploaded = st.file_uploader("Drag & drop a PNG/JPEG or browse files", type=["png", "jpg", "jpeg"])
    else:
        uploaded = st.camera_input("Take a photo")

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
    # Progress bar for image processing/inference (rendered where the button card was)
    progress_bar = (placeholder.progress(0) if 'placeholder' in locals() else st.progress(0))
    status_text = st.empty()
    try:
        # Prep session state once
        if st.session_state.get("img_bytes") is None:
            status_text.caption("Preparing image…")
            st.session_state["img_bytes"] = uploaded.getvalue()
            progress_bar.progress(10)
        if st.session_state.get("img_pil") is None:
            st.session_state["img_pil"] = img

        # Prepare bytes locally (avoid reading session_state from a thread)
        img_bytes_local = st.session_state["img_bytes"]

        # Run API call in a background thread so we can animate progress
        api_res_holder = {}
        api_err_holder = {}

        def _worker(api_url: str, payload: bytes, timeout_s: float):
            try:
                api_res_holder["res"] = call_api(api_url, payload, timeout_s=timeout_s)
            except Exception as _e:
                api_err_holder["err"] = _e

        status_text.caption("Contacting model API…")
        t = threading.Thread(target=_worker, args=(vit_api_url.strip(), img_bytes_local, float(request_timeout_s)), daemon=True)
        t.start()

        start = time.time()
        timeout_s = float(request_timeout_s)
        # Animate progress to 95% while waiting
        while t.is_alive():
            elapsed = time.time() - start
            pct = min(95, int((elapsed / timeout_s) * 95))
            progress_bar.progress(max(15, pct))
            time.sleep(0.1)

        # Thread finished; handle result
        if "err" in api_err_holder:
            raise api_err_holder["err"]

        progress_bar.progress(100)
        status_text.caption("Parsing results…")
        api_res = api_res_holder.get("res", {})
        top_species = str(api_res.get("species", "unknown"))
        top_prob = float(api_res.get("confidence", 0.0))
        api_edibility = api_res.get("edibility")  # may be 'edible'/'not edible'
        probs_series = pd.Series({top_species: top_prob}).sort_values(ascending=False)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        probs_series = pd.Series(dtype=float)
    finally:
        progress_bar.empty()
        status_text.empty()

    if not probs_series.empty:
        tab_res, tab_conf, tab_about, tab_map = st.tabs(["Result", "Confidence", "About", "Map"])

        with tab_res:
            if top_prob < conf_threshold:
                st.markdown('<span class="pill warn">Abstained</span>', unsafe_allow_html=True)
                st.error(f"Max confidence {top_prob:.2%} is below safety threshold {conf_threshold:.0%}.")
                st.caption("Following the safety gate rule to avoid risky advice.")
            else:
                # Prefer API-provided edibility when available; fallback to CSV
                if 'api_edibility' in locals() and api_edibility:
                    edibility = str(api_edibility).strip().title()
                    pill_cls = "warn" if edibility.lower().startswith("not") else ("ok" if edibility.lower().startswith("edib") else "unknown")
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
                    st.metric("Confidence src", "ViT API")

                if edibility == "Not Edible":
                    st.warning("⚠️ Do not consume wild mushrooms.")
                elif edibility == "Edible":
                    st.caption("Informational only. Never eat wild mushrooms based on an app prediction.")

        with tab_conf:
            st.markdown("**Confidence**")
            chart_df = probs_series.to_frame("probability")
            st.bar_chart(chart_df)
            st.dataframe(
                pd.DataFrame({
                    "species": chart_df.index,
                    "probability": chart_df["probability"].values
                }),
                use_container_width=True,
            )
            # Single API; comparison table removed

        with tab_about:
            st.markdown(
                """
                Frontend (Streamlit) → ViT API → JSON → Display
                Pipeline: Species → Edibility with a safety threshold (abstain)
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
