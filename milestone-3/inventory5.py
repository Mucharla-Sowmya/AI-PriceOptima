import pandas as pd
import numpy as np

# Load dataset (use your demand-features dataset)
df = pd.read_csv("demand.csv")

# Convert Date to proper format
df["Date"] = pd.to_datetime(df["Date"])

# Ensure numeric
df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")
df["Stock Level"] = pd.to_numeric(df["Stock Level"], errors="coerce")

# Sort for safe operations
df = df.sort_values(["Product ID", "Date"]).reset_index(drop=True)

# --------------------------------------------------------
# 1️⃣ INVENTORY RATIO
# --------------------------------------------------------

df["inventory_ratio"] = df["Stock Level"] / df["rolling_demand_7"]

# --------------------------------------------------------
# 2️⃣ DAYS UNTIL STOCK-OUT
# --------------------------------------------------------

df["days_until_stockout"] = df["Stock Level"] / df["rolling_demand_7"]

# Replace inf / NaN
df["days_until_stockout"] = df["days_until_stockout"].replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------------
# 3️⃣ LOW-STOCK & OVERSTOCK FLAGS
# --------------------------------------------------------

# Low stock if stock is less than 20% of weekly demand
df["low_stock"] = np.where(df["Stock Level"] < 0.2 * df["rolling_demand_7"], 1, 0)

# Over-stock if more than 3× 7-day demand
df["overstock"] = np.where(df["Stock Level"] > 3 * df["rolling_demand_7"], 1, 0)

# --------------------------------------------------------
# SAVE FINAL DATASET
# --------------------------------------------------------

df.to_csv("inventory.csv", index=False)

print("Inventory features added successfully!")
print(df[[
    "Product ID",
    "Date",
    "Stock Level",
    "rolling_demand_7",
    "inventory_ratio",
    "days_until_stockout",
    "low_stock",
    "overstock"
]].head(15))
