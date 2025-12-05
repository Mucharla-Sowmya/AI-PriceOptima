import pandas as pd
import numpy as np

# Load dataset with price, cost, units sold
df = pd.read_csv("inventory.csv")

# Ensure numeric fields
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

# --------------------------------------------------
# 1️⃣ Profit Per Unit
# --------------------------------------------------
df["profit_per_unit"] = df["Price"] - df["Cost"]

# --------------------------------------------------
# 2️⃣ Total Profit
# --------------------------------------------------
df["total_profit"] = df["Units Sold"] * df["profit_per_unit"]

# --------------------------------------------------
# 3️⃣ Profit Margin Percentage
# --------------------------------------------------
df["profit_margin_pct"] = np.where(
    df["Price"] > 0,
    (df["profit_per_unit"] / df["Price"]) * 100,
    np.nan
)

# --------------------------------------------------
# 4️⃣ Weighted Margin (Profit / Revenue)
# --------------------------------------------------
df["weighted_margin"] = np.where(
    df["Revenue"] > 0,
    df["total_profit"] / df["Revenue"],
    np.nan
)

# Save final dataset
df.to_csv("profit.csv", index=False)

print("Profit features created successfully!")
print(df[[
    "Product ID",
    "Date",
    "Price",
    "Cost",
    "Units Sold",
    "profit_per_unit",
    "total_profit",
    "profit_margin_pct",
    "weighted_margin"
]].head(10))
