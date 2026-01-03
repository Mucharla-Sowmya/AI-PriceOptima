from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ------------------------------------
# FastAPI App
# ------------------------------------
app = FastAPI(
    title="PriceOptima Pricing API",
    version="1.0",
    description="ML-based dynamic pricing service"
)

# ------------------------------------
# Load trained model
# ------------------------------------
model = joblib.load("best_pricing_model.pkl")

# ------------------------------------
# Input Schema (Minimal & Clean)
# ------------------------------------
class PricingInput(BaseModel):
    price: float = Field(..., example=49.99)
    stock_level: int = Field(..., example=1200)
    day_of_week: int = Field(..., example=6)
    is_weekend: int = Field(..., example=1)
    month: int = Field(..., example=12)

# ------------------------------------
# Root Endpoint (optional but clean)
# ------------------------------------
@app.get("/", tags=["Health"])
def home():
    return {"status": "PriceOptima API running"}

# ------------------------------------
# Prediction Endpoint
# ------------------------------------
@app.post(
    "/predict-price",
    tags=["Pricing"],
    summary="Get ML-based recommended price"
)
def predict_price(data: PricingInput):

    # Prepare base features
    base_features = np.array([
        data.price,
        data.stock_level,
        data.day_of_week,
        data.is_weekend,
        data.month
    ])

    # Match model input size
    full_features = np.zeros((1, model.n_features_in_))
    full_features[0, :len(base_features)] = base_features

    # Predict demand
    predicted_demand = model.predict(full_features)[0]

    # Pricing logic
    recommended_price = data.price
    if predicted_demand > 200:
        recommended_price *= 1.05
    elif predicted_demand < 50:
        recommended_price *= 0.95

    return {
        "predicted_demand": round(float(predicted_demand), 2),
        "recommended_price": round(float(recommended_price), 2)
    }
