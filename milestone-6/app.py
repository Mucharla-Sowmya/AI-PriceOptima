#uvicorn app:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ------------------------------------
# App Setup (Clear for Non-Technical Users)
# ------------------------------------
app = FastAPI(
    title="PriceOptima – Smart Pricing API",
    description="""
This API recommends an optimal product price based on:
• Current price  
• Inventory level  
• Day and month information  

Simply enter the details and get a smart price suggestion.
""",
    version="1.0"
)

# ------------------------------------
# Load trained ML model
# ------------------------------------
model = joblib.load("best_pricing_model.pkl")

# ------------------------------------
# Simple Input Form (Self-Explanatory)
# ------------------------------------
from pydantic import BaseModel, Field

class PricingInput(BaseModel):
    price: float = Field(
        ...,
        description="Current selling price of the product",
        example=49.99
    )

    stock_level: int = Field(
        ...,
        description="Available inventory units in stock",
        example=1200
    )

    day_of_week: int = Field(
        ...,
        description="Day of the week (0 = Monday, 6 = Sunday)",
        example=6
    )

    is_weekend: int = Field(
        ...,
        description="Is it a weekend? (1 = Yes, 0 = No)",
        example=1
    )

    month: int = Field(
        ...,
        description="Month number (1 = January, 12 = December)",
        example=12
    )

# ------------------------------------
# Home Page (Plain English)
# ------------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to PriceOptima Smart Pricing System",
        "how_to_use": "Open /docs and enter product details to get a recommended price"
    }

# ------------------------------------
# Price Recommendation Endpoint
# ------------------------------------
@app.post(
    "/predict-price",
    summary="Get Recommended Product Price",
    description="Returns predicted demand and an optimized selling price"
)
def predict_price(data: PricingInput):

    # Convert user input to model format
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

    # Simple pricing logic
    recommended_price = data.price
    if predicted_demand > 200:
        recommended_price *= 1.05   # High demand → increase price
    elif predicted_demand < 50:
        recommended_price *= 0.95   # Low demand → decrease price
    
    return {
        "predicted_demand": round(float(predicted_demand), 2),
        "recommended_price": round(float(recommended_price), 2)
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)