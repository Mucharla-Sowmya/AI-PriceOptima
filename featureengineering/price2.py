import pandas as pd
import numpy as np

# -----------------------------------
# Load dataset
# -----------------------------------
df = pd.read_csv("daily_fixed_dataset.csv")

# Parse dates
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

# Sort for lag logic
df = df.sort_values(by=["Product ID", "Date"]).reset_index(drop=True)

# Ensure numeric columns
num_cols = ["Price", "AvgPrice", "Cost"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------------
# 1. LAG FEATURES
# -----------------------------------
df["price_lag_1"] = df.groupby("Product ID")["Price"].shift(1)
df["price_lag_7"] = df.groupby("Product ID")["Price"].shift(7)

# -----------------------------------
# 2. PRICE CHANGE %
# -----------------------------------
df["price_change_pct"] = ((df["Price"] - df["price_lag_1"]) / df["price_lag_1"]) * 100

# Replace inf values caused by division by zero
df["price_change_pct"] = df["price_change_pct"].replace([np.inf, -np.inf], np.nan)

# -----------------------------------
# 3. DISCOUNT %
#     discount = (List Price - Selling Price) / List Price
# -----------------------------------
df["discount_pct"] = ((df["AvgPrice"] - df["Price"]) / df["AvgPrice"]) * 100
df["discount_pct"] = df["discount_pct"].replace([np.inf, -np.inf], np.nan)

# -----------------------------------
# 4. MARGIN %
#     margin = (Selling Price - Cost Price) / Cost Price
# -----------------------------------
df["margin_pct"] = ((df["Price"] - df["Cost"]) / df["Cost"]) * 100
df["margin_pct"] = df["margin_pct"].replace([np.inf, -np.inf], np.nan)

# -----------------------------------
# 5. Fill lag columns only NOW
#    Correct behavior: Keep NaN for price_change_pct
# -----------------------------------
df["price_lag_1"] = df["price_lag_1"].fillna(0)
df["price_lag_7"] = df["price_lag_7"].fillna(0)

# -----------------------------------
# Save output
# -----------------------------------
df.to_csv("price.csv", index=False)

print("FINAL price-based features generated correctly.\n")
print(df[["Price", "price_lag_1", "price_change_pct"]].head(15).to_string())
