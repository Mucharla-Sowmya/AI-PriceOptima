import pandas as pd
import numpy as np

# Load the daily fixed dataset
df = pd.read_csv("../milestone-1/combined_dataset.csv")

# Ensure Date is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Ensure numeric
df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")

# Sort before applying lags
df = df.sort_values(["Product ID", "Date"]).reset_index(drop=True)

# -----------------------------------------------------------
# STEP 1 — Calendar-Based Lag Features
# -----------------------------------------------------------

# Create helper table for daily values
daily = df[["Product ID", "Date", "Units Sold"]].copy()

# Create shifted-date columns
daily["Date_minus_1"] = daily["Date"] - pd.Timedelta(days=1)
daily["Date_minus_7"] = daily["Date"] - pd.Timedelta(days=7)
daily["Date_minus_30"] = daily["Date"] - pd.Timedelta(days=30)

# Lookup table
lookup = daily.rename(columns={"Units Sold": "DailyDemand"})[
    ["Product ID", "Date", "DailyDemand"]
]

# Merge lag-1
daily = daily.merge(
    lookup.rename(columns={"Date": "Date_minus_1", "DailyDemand": "demand_lag_1"}),
    on=["Product ID", "Date_minus_1"],
    how="left"
)

# Merge lag-7
daily = daily.merge(
    lookup.rename(columns={"Date": "Date_minus_7", "DailyDemand": "demand_lag_7"}),
    on=["Product ID", "Date_minus_7"],
    how="left"
)

# Merge lag-30
daily = daily.merge(
    lookup.rename(columns={"Date": "Date_minus_30", "DailyDemand": "demand_lag_30"}),
    on=["Product ID", "Date_minus_30"],
    how="left"
)

# -----------------------------------------------------------
# STEP 2 — Rolling Windows (Correct Calendar Order)
# -----------------------------------------------------------

daily = daily.sort_values(["Product ID", "Date"])

# Rolling mean 7 days
daily["rolling_demand_7"] = (
    daily.groupby("Product ID")["Units Sold"]
    .rolling(7, min_periods=1)
    .mean()
    .reset_index(0, drop=True)
)

# Rolling mean 30 days
daily["rolling_demand_30"] = (
    daily.groupby("Product ID")["Units Sold"]
    .rolling(30, min_periods=1)
    .mean()
    .reset_index(0, drop=True)
)

# Rolling std (volatility)
daily["volatility_7"] = (
    daily.groupby("Product ID")["Units Sold"]
    .rolling(7, min_periods=1)
    .std()
    .reset_index(0, drop=True)
)

daily["volatility_30"] = (
    daily.groupby("Product ID")["Units Sold"]
    .rolling(30, min_periods=1)
    .std()
    .reset_index(0, drop=True)
)

# -----------------------------------------------------------
# STEP 3 — Merge Features Back to Full Dataset
# -----------------------------------------------------------

final = df.merge(
    daily[
        [
            "Product ID",
            "Date",
            "demand_lag_1",
            "demand_lag_7",
            "demand_lag_30",
            "rolling_demand_7",
            "rolling_demand_30",
            "volatility_7",
            "volatility_30",
        ]
    ],
    on=["Product ID", "Date"],
    how="left"
)

# -----------------------------------------------------------
# Save Output
# -----------------------------------------------------------

final.to_csv("demand.csv", index=False)

print("Demand lag, rolling averages, and volatility features created successfully!")
print(final.head(10))  # prints first 10 rows
