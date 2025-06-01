from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from app.ml.predict import Predictor

app = FastAPI()

# Request schema
class TextRequest(BaseModel):
    text: str

# Root health check
@app.get("/")
def root():
    return {"message": "Text Classification API is running."}

# Prediction endpoint with optional version control
@app.post("/predict")
def predict_text(
    request: TextRequest,
    model_version: Optional[int] = Query(default=None, description="Model version to use."),
    encoder_version: Optional[int] = Query(default=None, description="Encoder version to use.")
):
    predictor = Predictor(model_version=model_version, encoder_version=encoder_version)
    prediction = predictor.predict(request.text)

    return {
        "prediction": prediction
    }