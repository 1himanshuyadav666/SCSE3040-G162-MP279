
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
    "shop_area",
    "block",
]

app = FastAPI(
    title="MP279 Ration Shop Grain Prediction API",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    cards_attached: float
    working_hours: float = Field(ge=1.5)
    biometric_failures: float
    temp_c: float = Field(le=80)
    stock_kg_morning: float
    days_to_month_end: float
    shop_area: str
    block: str


class PredictionOutput(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_data)[0]

    return {"prediction": float(prediction)}
