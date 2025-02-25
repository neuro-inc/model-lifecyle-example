import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI

from char_rnn_classification_tutorial import load_model, predict

from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    name: str = Field(..., description="Name to classify")
    n_predictions: int = Field(default=3, description="Number of classes to return")


class PredictionItem(BaseModel):
    value: float
    category: str


class PredictionResponse(BaseModel):
    prediction: List[PredictionItem]

MODEL_PATH = os.environ.get("MODEL_PATH", "models/weights.pt")
PARAMS_PATH = os.environ.get("PARAMS_PATH", "models/params.json")

model_params: Dict = None  # type: ignore
rnn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rnn, model_params
    # Load the ML model
    with open(PARAMS_PATH, "r") as f:
        model_params = json.load(f)
    rnn = load_model(Path(MODEL_PATH), model_params)
    logging.info("Model loaded")
    yield
    # Clean up the ML models and release the resources
    del rnn


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Service is up"}


@app.post("/predict")
def predict_text(request: PredictionRequest) -> PredictionResponse:
    prediction = predict(
        rnn=rnn,
        name=request.name,
        n_predictions=request.n_predictions,
        params=model_params,
    )
    prediction = [
        {"value": value.item(), "category": category} for value, category in prediction
    ]
    return {"prediction": prediction}  # type: ignore


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
