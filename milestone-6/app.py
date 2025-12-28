from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="PriceOptima Pricing API",
    description="ML-based dynamic pricing using LightGBM",
    version="1.0"
)
@app.get("/")
def root():
    return {
        "message": "Welcome to PriceOptima Dynamic Pricing API",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict-price"
        }
    }

# Load trained model
model = joblib.load("best_pricing_model.pkl")

class PricingInput(BaseModel):
    price: float = Field(..., gt=0, example=49.99)
    stock_level: int = Field(..., ge=0, example=1200)
    day_of_week: int = Field(..., ge=0, le=6, example=6)
    is_weekend: int = Field(..., ge=0, le=1, example=1)
    month: int = Field(..., ge=1, le=12, example=12)

@app.post("/predict-price")
def predict_price(data: PricingInput):

    base_features = np.array([
        data.price,
        data.stock_level,
        data.day_of_week,
        data.is_weekend,
        data.month
    ])

    full_features = np.zeros((1, model.n_features_in_))
    full_features[0, :len(base_features)] = base_features

    predicted_demand = model.predict(full_features)[0]

    recommended_price = data.price
    if predicted_demand > 200:
        recommended_price *= 1.05
    elif predicted_demand < 50:
        recommended_price *= 0.95

    return {
        "predicted_demand": round(float(predicted_demand), 2),
        "recommended_price": round(float(recommended_price), 2)
    }
