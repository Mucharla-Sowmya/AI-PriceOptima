import pandas as pd

# ------------------------------------------------------------
# STEP 1.1: Load cleaned & feature-engineered dataset
# ------------------------------------------------------------
df = pd.read_csv("../milestone-3/cleaned_features.csv")
# ------------------------------------------------------------
# STEP 1.2: Check missing values
# ------------------------------------------------------------
print("\n========== MISSING VALUES CHECK ==========")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 1.3: Handle missing values
# (Simple and acceptable approach for this milestone)
# ------------------------------------------------------------
df.fillna(0, inplace=True)

print("\nMissing values handled.")

# ------------------------------------------------------------
# STEP 1.5: Encode categorical variables
# (One-hot encoding)
# ------------------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

print("\n========== DATASET AFTER ENCODING ==========")
print(df_encoded.head())
print("\nEncoded Shape:", df_encoded.shape)


# ------------------------------------------------------------
# STEP 1.6: Define features (X) and target (y)
# Target variable = Units Sold (Demand)
# ------------------------------------------------------------
y = df_encoded['Units Sold']
X = df_encoded.drop(columns=['Units Sold'])

print("\n========== FINAL FEATURE SET ==========")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nSample X:")
print(X.head())

print("\nSample y:")
print(y.head())


# ------------------------------------------------------------
# STEP 1 COMPLETE
# ------------------------------------------------------------
print("\n✅ STEP 1 COMPLETE: Dataset ready for ML training")



# ------------------------------------------------------------
# STEP 2: TIME-BASED TRAIN-TEST SPLIT
# ------------------------------------------------------------

# Add Date back for sorting (important)
df_encoded['Date'] = pd.to_datetime(df['Date'])

# Sort dataset by time
df_encoded = df_encoded.sort_values('Date')

# Drop Date after sorting (not used as feature)
df_encoded = df_encoded.drop(columns=['Date'])

# Re-define X and y (after sorting)
y = df_encoded['Units Sold']
X = df_encoded.drop(columns=['Units Sold'])

# Define split index (80% train, 20% test)
split_index = int(len(df_encoded) * 0.8)

# Time-based split
X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("\n========== STEP 2: TIME-BASED TRAIN-TEST SPLIT ==========")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

print("\nTrain period (first & last rows):")
print(X_train.head(1))
print(X_train.tail(1))

print("\nTest period (first & last rows):")
print(X_test.head(1))
print(X_test.tail(1))

print("\n✅ STEP 2 COMPLETE: Time-based split successful")
# Clean feature names (recommended for LightGBM)
#X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
#X_test.columns  = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
# Create test dataframe for revenue & backtesting
df_test = df.iloc[X_test.index].copy()

#pip install numpy pandas scikit-learn xgboost lightgbm
print("\n========== STEP 3:  Train Machine Learning Models ==========")



from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("\n========== STEP 3A: XGBOOST TRAINING ==========")

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Metrics
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost MAE  : {xgb_mae:.4f}")
print(f"XGBoost RMSE : {xgb_rmse:.4f}")
from lightgbm import LGBMRegressor

print("\n========== STEP 3B: LIGHTGBM TRAINING ==========")

lgbm_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train
lgbm_model.fit(X_train, y_train)

# Predict
y_pred_lgbm = lgbm_model.predict(X_test)

# Metrics
lgbm_mae = mean_absolute_error(y_test, y_pred_lgbm)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))

print(f"LightGBM MAE  : {lgbm_mae:.4f}")
print(f"LightGBM RMSE : {lgbm_rmse:.4f}")
import joblib

# Save trained LightGBM model
joblib.dump(lgbm_model, "best_pricing_model.pkl")

print("✅ Model saved as best_pricing_model.pkl")

# ------------------------------------------------------------
# STEP 4: ML-BASED DYNAMIC PRICING SIMULATION
# ------------------------------------------------------------

print("\n========== STEP 4: ML-BASED PRICING SIMULATION ==========")

# Create a copy of test set for pricing simulation
pricing_df = X_test.copy()

# Add original values back (from original dataframe)
pricing_df['Price'] = df.loc[X_test.index, 'Price']
pricing_df['Units Sold'] = df.loc[X_test.index, 'Units Sold']

# Use LightGBM predicted demand
pricing_df['predicted_demand'] = y_pred_lgbm

# Define demand thresholds
high_demand_threshold = pricing_df['predicted_demand'].quantile(0.75)
low_demand_threshold  = pricing_df['predicted_demand'].quantile(0.25)

# Initialize ML-based price
pricing_df['ml_price'] = pricing_df['Price']

# Increase price for high demand (+5%)
pricing_df.loc[
    pricing_df['predicted_demand'] >= high_demand_threshold,
    'ml_price'
] *= 1.05

# Decrease price for low demand (-5%)
pricing_df.loc[
    pricing_df['predicted_demand'] <= low_demand_threshold,
    'ml_price'
] *= 0.95

# Show preview
print(pricing_df[['Price', 'predicted_demand', 'ml_price']].head(10))

print("\n✅ STEP 4 COMPLETE: ML-based prices generated")
#-----------------------------------step 5----------------------------------------------------------
# Align test dataframe with X_test
df_test = df.iloc[X_test.index].copy()
# Re-extract time features (required for rule-based pricing)
df_test['Date'] = pd.to_datetime(df_test['Date'], errors='coerce')

df_test['day_of_week'] = df_test['Date'].dt.dayofweek
df_test['month'] = df_test['Date'].dt.month
df_test['day'] = df_test['Date'].dt.day

# Attach ML prices from Step 4
df_test['ml_price'] = pricing_df['ml_price']
df_test['weekend_factor'] = df_test['day_of_week'].isin([5, 6]).astype(int)
df_test['weekend_factor'] = df_test['weekend_factor'].replace({1: 1.10, 0: 1.0})

df_test['season_factor'] = df_test['month'].apply(
    lambda x: 1.15 if x == 12 else 1.0
)

df_test['monthend_factor'] = df_test['day'].apply(
    lambda x: 1.05 if x >= 25 else 1.0
)

df_test['lowdemand_factor'] = df_test['month'].apply(
    lambda x: 0.95 if x in [2, 4] else 1.0
)
df_test['inventory_factor'] = 1.0

# Low stock
df_test.loc[df_test['Stock Level'] <= 2000, 'inventory_factor'] *= 1.10

# High stock
df_test.loc[
    (df_test['Stock Level'] > 3500) &
    (df_test['Stock Level'] <= 4500),
    'inventory_factor'
] *= 0.90

# Overstock
df_test.loc[df_test['Stock Level'] > 4500, 'inventory_factor'] *= 0.80
df_test['rule_price'] = (
    df_test['Price'] *
    df_test['weekend_factor'] *
    df_test['season_factor'] *
    df_test['monthend_factor'] *
    df_test['lowdemand_factor'] *
    df_test['inventory_factor']
)
print("\n========== STEP 5: REVENUE CALCULATION ==========")

df_test['static_revenue'] = df_test['Price'] * df_test['Units Sold']
df_test['rule_revenue']   = df_test['rule_price'] * df_test['Units Sold']
df_test['ml_revenue']     = df_test['ml_price'] * df_test['Units Sold']

print(df_test[['Price', 'rule_price', 'ml_price',
               'Units Sold',
               'static_revenue',
               'rule_revenue',
               'ml_revenue']].head())
total_static_revenue = df_test['static_revenue'].sum()
total_rule_revenue   = df_test['rule_revenue'].sum()
total_ml_revenue     = df_test['ml_revenue'].sum()
print("\n--- TOTAL REVENUE SUMMARY ---")
print(f"Static Pricing Total Revenue : ${total_static_revenue:,.2f}")
print(f"Rule-Based Pricing Total Revenue : ${total_rule_revenue:,.2f}")
print(f"ML-Based Pricing Total Revenue : ${total_ml_revenue:,.2f}")
revenue_lift = (
    (total_ml_revenue - total_static_revenue) /
    total_static_revenue
) * 100
print(f"\nML-Based Pricing Revenue Lift vs Static: {revenue_lift:.2f}%")
