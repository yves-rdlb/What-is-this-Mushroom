from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
<<<<<<< HEAD
from tensorflow import keras

# Import Model
from MUSHROOM.model_functions.save_load import load_model_gcs

# Import Preprocessing
from MUSHROOM.model_functions.functions import preprocess_for_predict

app = FastAPI()

# Load model
app.state.model = load_model_gcs()

=======
from pathlib import Path
import json
from tensorflow import keras
import numpy as np
import os

app = FastAPI()

>>>>>>> master
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Get current file's directory and go up 2 levels
CURRENT_DIR = Path(__file__).parent  # MUSHROOM/api/
base_dir = CURRENT_DIR.parent.parent  # Project root

model_path = base_dir / "model" / "mushroom_model_EfficientNetV2B0_6.keras"
label_path = base_dir / "model" / "class_indices.json"


app.state.model = keras.models.load_model(model_path)

# load labels + # Load label mapping
#label_path = base_dir / "model" / "class_indices.json"

with open(label_path, "r", encoding="utf-8") as f:
    map = json.load(f)
    app.state.class_labels = {int(v): k for k, v in map.items()}


# Import Preprocessing
# from MUSHROOM import



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
        image = Image.open(io.BytesIO(contents)).convert('RGB') # to match preprocessing
        image = image.resize((224,224))
        img=np.array(image).astype('float32') /255.0
        img=np.expand_dims(img, axis=0)
        
        # Run your model prediction
        model = app.state.model
        index_to_class = app.state.class_labels
        
        prediction=model.predict(img)
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
