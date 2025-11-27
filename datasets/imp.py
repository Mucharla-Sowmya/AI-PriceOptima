import pandas as pd
import numpy as np

# STEP 0 — LOAD DATA
file_path = "combined_mandatory_fields_dataset.csv"
df = pd.read_csv(file_path)

print("\n================= LOADED DATA =================")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)


# STEP 1 — VALIDATE MANDATORY FIELDS
required_fields = [
    "Date",
    "Product ID",
    "Units Sold",
    "Price",
    "Revenue",
    "Stock Level",
    "Restock_Date",
    "Warehouse/Store ID",
    "Cost"              # ← added because KPI uses real cost
]

missing_fields = [c for c in required_fields if c not in df.columns]

print("\n================= STEP 1: FIELD VALIDATION =================")
if missing_fields:
    print("Missing Mandatory Fields:", missing_fields)
else:
    print("All mandatory fields are present!")


# STEP 2 — DATA QUALITY CHECK
print("\n================= STEP 2: DATA QUALITY CHECK =================")

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Convert numeric columns correctly
num_cols = ["Units Sold", "Price", "Revenue", "Stock Level", "Cost"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing stock levels using median
if "Stock Level" in df.columns:
    median_stock = df["Stock Level"].median()
    df["Stock Level"] = df["Stock Level"].replace(0, np.nan)
    df["Stock Level"] = df["Stock Level"].fillna(median_stock)


# STEP 3 — KPI CALCULATION
print("\n================= STEP 3: KPI CALCULATION =================")

# KPI 1 — Total Revenue
total_revenue = df["Revenue"].sum()

# KPI 2 — Total COGS using REAL COST from inventory
df["COGS"] = df["Cost"] * df["Units Sold"]
total_cogs = df["COGS"].sum()

# Profit margin
profit_margin = ((total_revenue - total_cogs) / total_revenue) * 100 if total_revenue else 0

# KPI 3 — Inventory Turnover
avg_inventory = df["Stock Level"].mean()
inventory_turnover = total_cogs / avg_inventory if avg_inventory else 0

# KPI 4 — Average Selling Price
average_selling_price = df["Price"].mean()

# KPI 5 — Average Units Sold
average_units_sold = df["Units Sold"].mean()

# Create KPI table
kpi_summary = pd.DataFrame({
    "Total Revenue": [total_revenue],
    "Total COGS": [total_cogs],
    "Profit Margin (%)": [profit_margin],
    "Inventory Turnover": [inventory_turnover],
    "Average Selling Price": [average_selling_price],
    "Average Units Sold": [average_units_sold]
})

# Save KPI summary
kpi_file = "KPI_summary_corrected.csv"
kpi_summary.to_csv(kpi_file, index=False)

print("\nKPI Summary:")
print(kpi_summary)
print("\nKPI Summary saved as:", kpi_file)
