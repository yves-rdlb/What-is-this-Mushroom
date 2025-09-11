# What Is This Mushroom?

Safety-first mushroom identification demo. The system predicts a species from an image and maps it to a deterministic edibility label, with an explicit confidence gate that abstains below a threshold.

This repo contains:
- A Streamlit frontend (API-first in `UI/app_v3.py`)
- A FastAPI backend (ViT model in `MUSHROOM/api/VIT_API.py`, legacy EfficientNet in `MUSHROOM/api/fast.py`)
- Model utilities in `MUSHROOM/model_functions/`
- Lookup tables in `lookups/` and optional distribution data in `distributions/`


## How It Works

1) Image → Species (Model)
- The model receives an RGB image (resized to 224×224) and outputs a probability distribution over ~169 species.
- Current production path: ViT SavedModel (TensorFlow) served via FastAPI.

2) Species → Edibility (Deterministic)
- Predicted species is normalized and matched against `lookups/species_edibility.csv`.
- The CSV encodes edibility as 0 = Edible, 1 = Not Edible. If a species is missing, edibility is “Unknown”.

3) Safety Gate (Abstain)
- If the top class confidence is below a user-set threshold (default ~0.85), the app abstains from showing an edibility claim and explains why.

4) Display & Map (Optional)
- The Streamlit UI shows: top prediction, confidence, edibility label, and a bar chart of probabilities.
- A species heatmap can be shown using `distributions/species_points.csv` (lat, lon, optional month); this is illustrative only.


## Architecture

- Frontend: Streamlit app in `UI/app_v3.py`
  - API-only client. Calls one REST endpoint (`/predict`) and renders results.
  - Polished layout with a safety threshold control and optional heatmap (pydeck).

- Backend: FastAPI
  - ViT API (`MUSHROOM/api/VIT_API.py`):
    - Downloads a ViT SavedModel on first run via `load_vit_model()` and serves a `/predict` endpoint.
    - Returns JSON compatible with the UI, e.g. `{ species, confidence, edibility }`.
  - EfficientNet API (`MUSHROOM/api/fast.py`):
    - Legacy path loading a Keras model and using EfficientNetV2 preprocessing.
    - Note: the path currently points to `model_main/...`, which is not present in this repo. Prefer the ViT API, or fix the path before use.

- Lookups & Data
  - `lookups/species_edibility.csv`: species → edibility mapping used by the UI.
  - `lookups/species_classes.json`: class list used by older UIs; superseded by the model’s internal mapping in the ViT API.
  - `distributions/species_points.csv`: optional map data (sampled occurrences by species).


## Inference Pipeline (Technical)

- Preprocessing:
  - Images are read as RGB and resized to 224×224.
  - ViT path: normalized to [0,1] and passed into the SavedModel signature.
  - EfficientNet path: uses `efficientnet_v2.preprocess_input` (see `MUSHROOM/api/fast.py`).

- Model output:
  - ViT API returns top class and a confidence. Confidence is normalized to 0..1; percentage strings are parsed when necessary.
  - EfficientNet returns logits or probabilities; UI converts to a pandas Series for display.

- Post-processing:
  - Species key normalized to lowercase with underscores before CSV lookup.
  - Edibility label resolved deterministically: 0 → “Edible”, 1 → “Not Edible”, missing → “Unknown”.
  - Confidence gate enforces abstention below threshold.


## Models

- Vision Transformer (ViT, TensorFlow SavedModel)
  - Served by `MUSHROOM/api/VIT_API.py` using utilities in `MUSHROOM/model_functions/vit_model_functions.py`.
  - Weights are downloaded (once) from a GitHub release zip into `models/vit_saved_model/`.
  - API returns `{species, confidence, edibility}` and a legacy `prediction` object for compatibility.

- EfficientNetV2B0 (Keras .keras)
  - Legacy model; example artifacts in `model/` (e.g., `mushroom_model_EfficientNetV2B0_6.keras`).
  - The FastAPI server in `MUSHROOM/api/fast.py` uses EfficientNet preprocessing and expects a class index mapping to invert to species names.
  - Path currently references `model_main/...` which is not present; update to an existing `.keras` or prefer the ViT API.

- Class Set
  - ~169 target species. In some modules the `index_to_class` mapping is hardcoded; in others it comes from a JSON/CSV. Consolidation is an improvement item (see below).


## What We’ve Covered

- API-first UI pattern: Streamlit frontend calls a single `/predict` endpoint and adapts to both the “flat” and “legacy nested” JSON shapes.
- Safety gating: UI abstains on low confidence and clearly communicates why.
- Deterministic edibility mapping: CSV-driven mapping separated from the model.
- Distribution visualization: Optional heatmap and dot overlay per species, filtered by month.
- Health checks: `scripts/health_check_api.py` verifies the API root and predict endpoints.
- Dockerization: Two Dockerfiles — EfficientNet (`Dockerfile`) and ViT (`Dockerfile_vit`).
- Make targets: `run_vit_api`, `run_api`, `api_health`, `streamlit` and Docker-related tasks.


## Areas for Improvement

- Unify class mappings
  - Replace hardcoded `index_to_class` dicts with a single source (e.g., `lookups/species_classes.json`) to avoid drift.

- Fix legacy EfficientNet paths
  - `MUSHROOM/api/fast.py` expects `model_main/...`; either update to an existing file under `model/` or remove in favor of the ViT API to reduce confusion.

- Confidence calibration
  - Consider temperature scaling or isotonic regression for better-calibrated probabilities, improving the abstain threshold’s behavior.

- Edibility coverage and provenance
  - Expand and document the edibility CSV, including sources/attribution and ambiguity handling. Add unit tests around normalization and lookup.

- API schema stability
  - Commit to a single, stable response shape `{species, confidence, edibility}` and phase out the legacy `prediction` wrapper after the UI migration is complete.

- Caching and cold start
  - Persist the downloaded ViT model in container images or volumes for faster cold starts; add retry/backoff on initial download.

- Testing & CI
  - Add lightweight tests for image preprocessing, CSV lookups, and API contracts; enforce with CI.

- Safety UX
  - Provide explicit warnings for lookalikes and add “Unknown” handling in more places (e.g., map tab) when abstaining.


## Quickstart

Local API (ViT):

```
pip install -r requirements.txt
uvicorn MUSHROOM.api.VIT_API:app --reload --port 8000
# or: make run_vit_api
```

Frontend:

```
pip install -r requirements_app.txt
streamlit run UI/app_v3.py
# Set endpoint to http://127.0.0.1:8000/predict/
```

Health check:

```
python scripts/health_check_api.py --url http://127.0.0.1:8000/predict
```

Docker (ViT):

```
docker build -f Dockerfile_vit -t mushroom-vit:local .
docker run -p 8000:8000 -e PORT=8000 mushroom-vit:local
```


## Repo Structure

```
MUSHROOM/
  api/
    VIT_API.py         # ViT FastAPI
    fast.py            # EfficientNet FastAPI (legacy)
  model_functions/
    vit_model_functions.py
    functions.py       # legacy utilities
  UI/
    yves_app.py        # simple client (legacy)

UI/
  app_v3.py            # API-first Streamlit UI (current)
  app_v1.py, app_v2.py # earlier lessons/variants

lookups/
  species_edibility.csv
  species_classes.json

distributions/
  species_points.csv   # optional heatmap data

scripts/
  health_check_api.py  # GET /, POST /predict
  make_classes_json.py # helper to build classes JSON from CSV
```


## Safety Disclaimer

This application is for demonstration and educational use only. Do not use it to make foraging or consumption decisions. Always consult local experts and authoritative resources.
