from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from pathlib import Path
import json
from tensorflow import keras
import numpy as np
import os
from MUSHROOM.model_functions.vit_model_functions import load_vit_model,vit_preprocess_for_predict,vit_predict
import tensorflow as tf



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

model_path = "https://github.com/yves-rdlb/What-is-this-Mushroom/releases/download/vit_saved_model_v0/vit_saved_model.zip"

# print(CURRENT_DIR)
# print(base_dir)
# print(model_path)
# Load model
app.state.model = load_vit_model(url=model_path)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file as bytes
        contents = await file.read()

        # Convert to PIL Image
        image=vit_preprocess_for_predict(contents,binary=True)

        # Run your model prediction
        model = app.state.model

        prediction=vit_predict(model,image)
        mushroom,edibility,proba=prediction
        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": {
                               'class': mushroom,
                               'edibility': edibility,
                               'confidence': proba}})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
