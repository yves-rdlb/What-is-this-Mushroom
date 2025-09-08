from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
#from pathlib import Path
import json
from tensorflow import keras
import numpy as np
<<<<<<< HEAD
=======
import os
<<<<<<< HEAD
=======
>>>>>>> 8ccca9c0f2f0b84000eee25804730302e29eb6c9

>>>>>>> 1c69bdabb1fbf81349d95d0149fbd883dcb20fb5
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from MUSHROOM.model_functions.vit_model_functions import load_vit_model,vit_preprocess_for_predict,vit_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# Resolve project root relative to this file (MUSHROOM/api/.. -> repo root)
CURRENT_DIR = Path(__file__).resolve().parent
base_dir = CURRENT_DIR.parent.parent
=======
# Get current file's directory and go up 2 levels
CURRENT_DIR = os.path.dirname(__file__)  # MUSHROOM/api/
base_dir = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Project root
>>>>>>> 8ccca9c0f2f0b84000eee25804730302e29eb6c9

model_path = os.path.join(base_dir, "model_main", "mushroom_model_EfficientNetV2B0_finetuned.keras")
label_path = os.path.join(base_dir, "model_main", "class_indices.json")

<<<<<<< HEAD
# Load model_1
app.state.model_1 = keras.models.load_model(model_path)

# Load model_2
vit16_model_path = "https://github.com/yves-rdlb/What-is-this-Mushroom/releases/download/vit_saved_model_v0/vit_saved_model.zip"
app.state.model_2 = load_vit_model(url=vit16_model_path)
=======
<<<<<<< HEAD
# Load model once at startup
=======
# print(CURRENT_DIR)
# print(base_dir)
# print(model_path)
# Load model
>>>>>>> 8ccca9c0f2f0b84000eee25804730302e29eb6c9
app.state.model = keras.models.load_model(model_path)
>>>>>>> 1c69bdabb1fbf81349d95d0149fbd883dcb20fb5

# Load label mapping once at startup (name -> index), then invert to index -> name
with open(label_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)
    app.state.class_labels = {int(v): k for k, v in mapping.items()}


@app.get("/")
def root():
    return {"message": "API is running"}


# Accept both /predict and /predict/ for convenience
@app.post("/predict")
@app.post("/predict/")
async def predict(file: UploadFile=File(...)):
    try:
        # Read file as bytes
        contents = await file.read()

<<<<<<< HEAD
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224,224))
        image=np.expand_dims(image, axis=0)

        # Preprocessing
        image = preprocess_input(image)

        # Run your model prediction
        model_1 = app.state.model_1
        index_to_class = app.state.class_labels
        prediction=model_1.predict(image)
        index=int(np.argmax(prediction[0]))
        proba = round(np.max(prediction[0]) * 100, 2)
        mushroom=index_to_class.get(index, f"class_{index}")

        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": {
                               'class': mushroom,
                               'index': index,
                               'confidence': proba}})
=======
        # Convert to PIL Image and preprocess
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Run model prediction
        model = app.state.model
        index_to_class = app.state.class_labels
        prediction = model.predict(image)
        index = int(np.argmax(prediction[0]))
        prob = float(np.max(prediction[0]))  # 0..1
        species = index_to_class.get(index, f"class_{index}")

        # Match UI/app_v3.py expected contract: {"species": str, "confidence": float 0..1}
        return JSONResponse(content={"species": species, "confidence": prob})
>>>>>>> 1c69bdabb1fbf81349d95d0149fbd883dcb20fb5
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict_2/")
async def predict_2(file: UploadFile=File(...)):
    try:
        # Read file as bytes
        contents = await file.read()

        # Convert to PIL Image & Preprocess
        image=vit_preprocess_for_predict(contents, binary=True)

        # Run your model prediction
        model_2 = app.state.model_2
        prediction = vit_predict(model_2, image)
        mushroom, edibility, proba = prediction

        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": {
                               'class': mushroom,
                               'edibility': edibility,
                               'confidence': proba}})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
