# ============================================================
# FEATURE ENGINEERING FOR DYNAMIC PRICING & DEMAND PREDICTION
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# STEP 0: LOAD DATA & BASIC CLEANING
# ------------------------------------------------------------

# Load dataset
df = pd.read_csv("../milestone-1/combined_dataset.csv")

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sort data by date (important for lag features)
df = df.sort_values('Date')

# Remove duplicate rows
df = df.drop_duplicates()

# ------------------------------------------------------------
# STEP 1: TIME-BASED FEATURES
# ------------------------------------------------------------

df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek

# Weekend indicator (Saturday=5, Sunday=6)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Season feature (India-relevant)
df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Summer', 4: 'Summer', 5: 'Summer',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon',
    9: 'Post-Monsoon', 10: 'Post-Monsoon', 11: 'Post-Monsoon'
})

# Festival / holiday indicator (assumed for Oct–Dec)
df['is_festival'] = df['month'].isin([10, 11, 12]).astype(int)

# ------------------------------------------------------------
# STEP 2: PRICE-BASED FEATURES
# ------------------------------------------------------------

# Lag prices
df['price_lag_1'] = df['Price'].shift(1)
df['price_lag_7'] = df['Price'].shift(7)

# Percentage price change
df['price_change_pct'] = (df['Price'] - df['price_lag_1']) / df['price_lag_1']

# Discount percentage (bounded)
df['discount_pct'] = df['price_change_pct'].clip(lower=-1, upper=1)

# ------------------------------------------------------------
# STEP 3: DEMAND FEATURES (LAG & ROLLING)
# ------------------------------------------------------------

# Lag demand (Units Sold)
df['sales_lag_1'] = df['Units Sold'].shift(1)
df['sales_lag_7'] = df['Units Sold'].shift(7)
df['sales_lag_30'] = df['Units Sold'].shift(30)

# Rolling averages
df['rolling_sales_7'] = df['Units Sold'].rolling(window=7).mean()
df['rolling_sales_30'] = df['Units Sold'].rolling(window=30).mean()

# Demand volatility (standard deviation)
df['demand_volatility'] = df['Units Sold'].rolling(window=7).std()

# ------------------------------------------------------------
# STEP 4: PRICE ELASTICITY FEATURES
# ------------------------------------------------------------

# Percentage changes
df['price_pct_change'] = df['Price'].pct_change()
df['demand_pct_change'] = df['Units Sold'].pct_change()

# Price elasticity calculation
df['price_elasticity'] = df['demand_pct_change'] / df['price_pct_change']

# Elasticity classification
df['elasticity_class'] = pd.cut(
    df['price_elasticity'],
    bins=[-np.inf, -1, -0.3, np.inf],
    labels=['High Elastic', 'Medium Elastic', 'Low Elastic']
)

# ------------------------------------------------------------
# STEP 5: INVENTORY FEATURES
# ------------------------------------------------------------

# Inventory to demand ratio
df['inventory_ratio'] = df['Stock Level'] / (df['Units Sold'] + 1)

# Days until stock-out (using rolling demand)
df['days_to_stockout'] = df['Stock Level'] / (df['rolling_sales_7'] + 1)

# Inventory condition flags
df['low_stock_flag'] = (df['Stock Level'] <= 10).astype(int)
df['overstock_flag'] = (df['Stock Level'] > 4000).astype(int)

# ------------------------------------------------------------
# STEP 6: PROFIT FEATURES
# ------------------------------------------------------------

# Assume cost is 70% of selling price
COST_FACTOR = 0.7

df['cost_price'] = df['Price'] * COST_FACTOR
df['profit_per_unit'] = df['Price'] - df['cost_price']
df['profit_margin'] = df['profit_per_unit'] / df['Price']

# ------------------------------------------------------------
# STEP 7: INTERACTION FEATURES
# ------------------------------------------------------------

df['weekend_price_interaction'] = df['is_weekend'] * df['Price']
df['season_discount_interaction'] = df['is_festival'] * df['discount_pct']
df['inventory_price_interaction'] = df['inventory_ratio'] * df['Price']

# ------------------------------------------------------------
# STEP 8: CATEGORICAL ENCODING (IF COLUMNS EXIST)
# ------------------------------------------------------------

categorical_cols = ['Product ID', 'Category', 'Brand', 'Store ID']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Encode season separately
df['season'] = df['season'].astype('category').cat.codes

# ------------------------------------------------------------
# STEP 9: FINAL CLEANING (SAFE VERSION)
# ------------------------------------------------------------

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Forward fill first
df.ffill(inplace=True)

# Fill remaining NaNs ONLY for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)
# ------------------------------------------------------------
# STEP 10: SAVE CLEANED FEATURE DATASET
# ------------------------------------------------------------

output_path = "../milestone-3/cleaned_features.csv"

df.to_csv(output_path, index=False)

print(f"Cleaned feature dataset saved successfully at: {output_path}")

# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

print("Feature Engineering Completed Successfully ✅")
print("Final Dataset Shape:", df.shape)
print(df.head())
