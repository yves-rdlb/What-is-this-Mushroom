from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from tensorflow import keras

# Import Model
from MUSHROOM.model_functions.save_load import load_model_gcs

# Import Preprocessing
from MUSHROOM.model_functions.functions import preprocess_for_predict

app = FastAPI()

# Load model
app.state.model = load_model_gcs()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
        image = Image.open(io.BytesIO(contents))

        # Run your model prediction
        prediction = predict_mushroom(image)

        return JSONResponse(content={"filename": file.filename, "prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(input_one: float,
            input_two: float):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }
