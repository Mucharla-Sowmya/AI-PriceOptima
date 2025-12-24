import pandas as pd
import numpy as np

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("profit.csv")

# ============================================================
# 1️⃣ Clean & Prepare Base Columns
# ============================================================

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Weekend flag
df["day_of_week"] = df["Date"].dt.dayofweek          # Mon=0 ... Sun=6
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Ensure numeric (important for interactions)
for col in ["Price", "inventory_ratio", "days_until_stockout"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ============================================================
# 2️⃣ Interaction Features Based on Existing Columns
# ============================================================

# Weekend × Price
df["weekend_price"] = df["is_weekend"] * df["Price"]

# Inventory × Price
df["inventory_price"] = df["inventory_ratio"] * df["Price"]

# Stockout × Price
df["stockout_price"] = df["days_until_stockout"] * df["Price"]

# ============================================================
# Save Output
# ============================================================
df.to_csv("interaction.csv", index=False)

print("🎉 Interaction features created based on available columns!")

# ============================================================
# Preview useful columns
# ============================================================
preview_cols = [
    "Product ID", "Date", "Price", "day_of_week", "is_weekend",
    "inventory_ratio", "days_until_stockout",
    "weekend_price", "inventory_price", "stockout_price"
]

preview_cols = [c for c in preview_cols if c in df.columns]

print(df[preview_cols].head(10))
