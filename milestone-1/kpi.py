import pandas as pd
import numpy as np

# =========================================================
# 1. LOAD DATASET
# =========================================================
df = pd.read_csv("combined_dataset.csv")
print("✅ Dataset Loaded:", df.shape)

# =========================================================
# 2. STANDARDIZE COLUMN NAMES
# =========================================================
df = df.rename(columns={
    "Date": "date",
    "Product ID": "product_id",
    "Units Sold": "units_sold",
    "Price": "price",
    "Revenue": "revenue",
    "Store ID": "store_id",
    "Stock Level": "stock",
    "Restock_Date": "restock_date",
    "Cost": "cost_price",
    "AvgPrice": "avg_price"
})

# Convert dates (warning-free)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["restock_date"] = pd.to_datetime(df["restock_date"], dayfirst=True, errors="coerce")

print("✅ Columns standardized")

# =========================================================
# 3. DATA VALIDATION
# =========================================================
required_columns = [
    "date", "product_id", "units_sold", "price",
    "revenue", "stock", "cost_price", "avg_price"
]

missing_cols = [c for c in required_columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

print("✅ Required columns present")

# =========================================================
# 4. DATA QUALITY CHECKS
# =========================================================
df = df.drop_duplicates()

df = df[
    (df["price"] > 0) &
    (df["units_sold"] >= 0) &
    (df["stock"] >= 0)
]

df = df[df["units_sold"] <= df["stock"]]

# Fill missing values safely
df[required_columns] = df[required_columns].ffill()

print("✅ Data quality checks completed")

# =========================================================
# 5. BASELINE METRICS
# =========================================================
df["baseline_profit"] = (df["price"] - df["cost_price"]) * df["units_sold"]

baseline_revenue = df["revenue"].sum()
baseline_profit = df["baseline_profit"].sum()

df["baseline_conversion"] = df["units_sold"] / (df["stock"] + 1)

baseline_inventory_turnover = (
    df.groupby("product_id")["units_sold"].sum()
    / df.groupby("product_id")["stock"].mean()
).mean()

# =========================================================
# 6. SMART DYNAMIC PRICING SIMULATION
# =========================================================
np.random.seed(42)

df["demand_ratio"] = df["units_sold"] / (df["stock"] + 1)

df["dynamic_price"] = np.where(
    df["demand_ratio"] >= df["demand_ratio"].median(),
    df["avg_price"] * np.random.uniform(1.05, 1.09, len(df)),
    df["avg_price"] * np.random.uniform(0.97, 1.01, len(df))
)

elasticity = np.where(
    df["demand_ratio"] >= df["demand_ratio"].median(),
    -0.3,
    -0.9
)

price_change_pct = (df["dynamic_price"] - df["avg_price"]) / df["avg_price"]

df["dynamic_units_sold"] = df["units_sold"] * (1 + elasticity * price_change_pct)
df["dynamic_units_sold"] = df["dynamic_units_sold"].clip(
    lower=df["units_sold"] * 0.95
)

df["dynamic_revenue"] = df["dynamic_price"] * df["dynamic_units_sold"]
df["dynamic_profit"] = (
    (df["dynamic_price"] - df["cost_price"]) * df["dynamic_units_sold"]
)

# =========================================================
# 7. KPI CALCULATIONS
# =========================================================
revenue_lift = ((df["dynamic_revenue"].sum() - baseline_revenue) / baseline_revenue) * 100
profit_margin_improvement = ((df["dynamic_profit"].sum() - baseline_profit) / baseline_profit) * 100

df["dynamic_conversion"] = df["dynamic_units_sold"] / (df["stock"] + 1)

conversion_baseline = df["baseline_conversion"].mean()
conversion_dynamic = df["dynamic_conversion"].mean()

dynamic_inventory_turnover = (
    df.groupby("product_id")["dynamic_units_sold"].sum()
    / df.groupby("product_id")["stock"].mean()
).mean()

# =========================================================
# 8. KPI SUMMARY (FORMATTED % OUTPUT)
# =========================================================
kpi_summary = pd.DataFrame({
    "KPI": [
        "Revenue Lift",
        "Profit Margin Improvement",
        "Conversion Rate (Baseline)",
        "Conversion Rate (Dynamic)",
        "Inventory Turnover (Baseline)",
        "Inventory Turnover (Dynamic)"
    ],
    "Value": [
        f"{round(revenue_lift, 2)}%",
        f"{round(profit_margin_improvement, 2)}%",
        f"{round(conversion_baseline * 100, 2)}%",
        f"{round(conversion_dynamic * 100, 2)}%",
        round(baseline_inventory_turnover, 2),
        round(dynamic_inventory_turnover, 2)
    ]
})

kpi_summary.to_csv("kpi_summary.csv", index=False)

# =========================================================
# 9. FINAL OUTPUT
# =========================================================
print("\n✅ KPI CALCULATION COMPLETED SUCCESSFULLY\n")
print(kpi_summary)
