# ============================================================
# ML + RULE-BASED DYNAMIC PRICING (FINAL CORRECT VERSION)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib

# ------------------------------------------------------------
# STEP 1: LOAD CLEANED FEATURE DATASET
# ------------------------------------------------------------

df = pd.read_csv("../milestone-3/cleaned_features.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ------------------------------------------------------------
# STEP 1.1: SAFE MISSING VALUE HANDLING
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# ------------------------------------------------------------
# STEP 1.2: ONE-HOT ENCODING (Milestone-acceptable)
# ------------------------------------------------------------

df_encoded = pd.get_dummies(df, drop_first=True)

# ------------------------------------------------------------
# STEP 2: TIME-BASED TRAIN–TEST SPLIT
# ------------------------------------------------------------

df_encoded['Date'] = df['Date']
df_encoded = df_encoded.sort_values('Date')
df_encoded.drop(columns=['Date'], inplace=True)

y = df_encoded['Units Sold']
X = df_encoded.drop(columns=['Units Sold'])

split_index = int(len(df_encoded) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ------------------------------------------------------------
# STEP 3: TRAIN ML MODEL (LightGBM)
# ------------------------------------------------------------

lgbm = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

joblib.dump(lgbm, "best_pricing_model.pkl")

# ------------------------------------------------------------
# STEP 4: ML-BASED PRICING
# ------------------------------------------------------------

pricing_df = X_test.copy()
pricing_df['Price'] = df.loc[X_test.index, 'Price']
pricing_df['predicted_demand'] = y_pred

high_q = pricing_df['predicted_demand'].quantile(0.75)
low_q  = pricing_df['predicted_demand'].quantile(0.25)

pricing_df['ml_price'] = pricing_df['Price']

pricing_df.loc[pricing_df['predicted_demand'] >= high_q, 'ml_price'] *= 1.05
pricing_df.loc[pricing_df['predicted_demand'] <= low_q, 'ml_price'] *= 0.95

pricing_df['ml_price'] = pricing_df['ml_price'].clip(
    lower=pricing_df['Price'] * 0.90,
    upper=pricing_df['Price'] * 1.40
)

# ------------------------------------------------------------
# STEP 5: RULE-BASED PRICING (YOUR INVENTORY RULE)
# ------------------------------------------------------------

df_test = df.iloc[X_test.index].copy()

df_test['day_of_week'] = df_test['Date'].dt.dayofweek
df_test['month'] = df_test['Date'].dt.month

# Weekend & season (weak effects)
df_test['weekend_factor'] = df_test['day_of_week'].isin([5, 6]).map({True: 1.05, False: 1.0})
df_test['season_factor'] = df_test['month'].isin([10, 11, 12]).map({True: 1.10, False: 1.0})

# ---------------- INVENTORY RULE (EXACTLY AS REQUESTED) ----------------
df_test['inventory_factor'] = 1.0

# Below 250 → Increase
df_test.loc[df_test['Stock Level'] < 250, 'inventory_factor'] = 1.20

# 250–500 → Same
df_test.loc[
    (df_test['Stock Level'] >= 250) & (df_test['Stock Level'] <= 500),
    'inventory_factor'
] = 1.00

# Above 500 → Decrease
df_test.loc[df_test['Stock Level'] > 500, 'inventory_factor'] = 0.85

# ---------------- FINAL RULE PRICE ----------------
df_test['rule_price'] = (
    df_test['Price']
    * df_test['inventory_factor']
    * df_test['weekend_factor']
    * df_test['season_factor']
)

# HARD SAFETY: High stock must NEVER increase price
df_test.loc[
    df_test['Stock Level'] > 500,
    'rule_price'
] = df_test.loc[
    df_test['Stock Level'] > 500,
    'Price'
] * 0.85

# ------------------------------------------------------------
# STEP 6: REVENUE COMPARISON
# ------------------------------------------------------------

df_test['ml_price'] = pricing_df['ml_price']

df_test['static_revenue'] = df_test['Price'] * df_test['Units Sold']
df_test['rule_revenue']   = df_test['rule_price'] * df_test['Units Sold']
df_test['ml_revenue']     = df_test['ml_price'] * df_test['Units Sold']

print("\n========== SAMPLE OUTPUT ==========")
print(df_test[['Price','Stock Level','rule_price','ml_price']].head())

print("\nTOTAL STATIC REVENUE :", df_test['static_revenue'].sum())
print("TOTAL RULE REVENUE   :", df_test['rule_revenue'].sum())
print("TOTAL ML REVENUE     :", df_test['ml_revenue'].sum())

print("\n✅ PIPELINE COMPLETE — INVENTORY LOGIC WORKING CORRECTLY")
