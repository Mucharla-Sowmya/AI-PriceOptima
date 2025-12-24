import pandas as pd

# Load dataset
df = pd.read_csv("../milestone-1/combined_dataset.csv")

#------------------------------------------------------------
# STEP 0: Convert Date Columns & Extract Features
#------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print("\n========== STEP 0: Date Features Extracted ==========")
print(df[['Date', 'day_of_week', 'is_weekend', 'month', 'day']].head(15))


#------------------------------------------------------------
# STEP 1: TIME-BASED PRICING FACTORS
#------------------------------------------------------------
df['weekend_factor'] = df['is_weekend'].apply(lambda x: 1.10 if x == 1 else 1.0)
df['season_factor'] = df['month'].apply(lambda x: 1.15 if x == 12 else 1.0)
df['monthend_factor'] = df['day'].apply(lambda x: 1.05 if x >= 25 else 1.0)
df['lowdemand_factor'] = df['month'].apply(lambda x: 0.95 if x in [2, 4] else 1.0)
df['time_based_price'] = (
    df['Price'] *
    df['weekend_factor'] *
    df['season_factor'] *
    df['monthend_factor'] *
    df['lowdemand_factor']
)

print("\n========== STEP 1: Time-Based Pricing Factors ==========")
print(df[['Date', 'Price', 'weekend_factor', 'season_factor', 'monthend_factor', 'lowdemand_factor', 'time_based_price']].head(15))


#------------------------------------------------------------
# STEP 2: INVENTORY-BASED PRICING FACTORS (Adjusted)
#------------------------------------------------------------
df['inventory_factor'] = 1.0

# LOW STOCK (≤2000) → +10%
df.loc[df['Stock Level'] <= 2000, 'inventory_factor'] *= 1.10

# HIGH STOCK (3500–4500) → -10%
df.loc[(df['Stock Level'] > 3500) & (df['Stock Level'] <= 4500), 'inventory_factor'] *= 0.90

# OVERSTOCK (>4500) → -20%
df.loc[df['Stock Level'] > 4500, 'inventory_factor'] *= 0.80

# 🔹 Price after inventory adjustment
df['inventory_based_price'] = df['Price'] * df['inventory_factor']

print("\n========== STEP 2: Inventory-Based Pricing (Price Impact) ==========")

print("\n--- LOW STOCK (≤2000) Examples ---")
print(df[df['Stock Level'] <= 2000][
    ['Date','Price','Stock Level','inventory_factor','inventory_based_price']
].head())

print("\n--- MEDIUM STOCK (2000–3500) Examples ---")
print(df[(df['Stock Level'] > 2000) & (df['Stock Level'] <= 3500)][
    ['Date','Price','Stock Level','inventory_factor','inventory_based_price']
].head())

print("\n--- HIGH STOCK (3500–4500) Examples ---")
print(df[(df['Stock Level'] > 3500) & (df['Stock Level'] <= 4500)][
    ['Date','Price','Stock Level','inventory_factor','inventory_based_price']
].head())

print("\n--- OVERSTOCK (>4500) Examples ---")
print(df[df['Stock Level'] > 4500][
    ['Date','Price','Stock Level','inventory_factor','inventory_based_price']
].head())


#------------------------------------------------------------
# STEP 3: FINAL RULE-BASED PRICE
#------------------------------------------------------------
df['rule_price'] = (
    df['Price'] *
    df['weekend_factor'] *
    df['season_factor'] *
    df['monthend_factor'] *
    df['lowdemand_factor'] *
    df['inventory_factor']
)

print("\n========== STEP 3: FINAL RULE-BASED PRICE (Time + Inventory Effects) ==========")
print(df[['Date', 'Price', 'Stock Level', 'rule_price']].head(20))

#-------------------------------------------------------
# STEP 4: REVENUE CALCULATION
#-------------------------------------------------------

print("\n========== STEP 4: Revenue Calculations ==========")

# Static Revenue (Original Pricing)
df['static_revenue'] = df['Price'] * df['Units Sold']

# Rule-Based Revenue (Dynamic Pricing)
df['rule_revenue'] = df['rule_price'] * df['Units Sold']

# Show preview
print(df[['Date', 'Price', 'rule_price', 'Units Sold', 
          'static_revenue', 'rule_revenue']].head(20))

#-------------------------------------------------------
# STEP 5: Revenue Lift Calculation
#-------------------------------------------------------

print("\n========== STEP 5: Revenue Lift ==========")

df['revenue_lift'] = df['rule_revenue'] - df['static_revenue']

# Show first 20 rows
print(df[['Date', 'static_revenue', 'rule_revenue', 'revenue_lift']].head(20))

# Total revenue comparison
total_static = df['static_revenue'].sum()
total_rule = df['rule_revenue'].sum()
total_lift = total_rule - total_static

print("\n--- TOTAL REVENUE COMPARISON ---")
print(f"Total Static Revenue: {total_static:,.2f}")
print(f"Total Rule-Based Revenue: {total_rule:,.2f}")
print(f"Total Revenue Lift: {total_lift:,.2f}")


if total_lift > 0:
    print("\n🎉 Positive Revenue Lift → Rule-based engine IMPROVED revenue!")
else:
    print("\n⚠️ Negative Revenue Lift → Rule-based engine REDUCED revenue.")
