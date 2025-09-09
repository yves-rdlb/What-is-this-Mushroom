from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from MUSHROOM.model_functions.vit_model_functions import load_vit_model,vit_preprocess_for_predict,vit_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
model_path = "https://github.com/yves-rdlb/What-is-this-Mushroom/releases/download/vit_saved_model_v0/vit_saved_model.zip"
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
