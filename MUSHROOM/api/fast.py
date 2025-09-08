from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
#from pathlib import Path
import json
from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from MUSHROOM.model_functions.vit_model_functions import load_vit_model,vit_preprocess_for_predict,vit_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get current file's directory and go up 2 levels
CURRENT_DIR = os.path.dirname(__file__)  # MUSHROOM/api/
base_dir = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Project root

model_path = os.path.join(base_dir, "model_main", "mushroom_model_EfficientNetV2B0_finetuned.keras")
label_path = os.path.join(base_dir, "model_main", "class_indices.json")

# Load model_1
app.state.model_1 = keras.models.load_model(model_path)

# Load model_2
vit16_model_path = "https://github.com/yves-rdlb/What-is-this-Mushroom/releases/download/vit_saved_model_v0/vit_saved_model.zip"
app.state.model_2 = load_vit_model(url=vit16_model_path)

# load labels + # Load label mapping
with open(label_path, "r", encoding="utf-8") as f:
    map = json.load(f)
    app.state.class_labels = {int(v): k for k, v in map.items()}

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

@app.post("/predict/")
async def predict(file: UploadFile=File(...)):
    try:
        # Read file as bytes
        contents = await file.read()

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
