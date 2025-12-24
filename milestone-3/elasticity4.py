import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset (the one with daily fixed + demand features)
df = pd.read_csv("../milestone-1/combined_dataset.csv")

# Ensure proper sorting
df = df.sort_values(["Product ID", "Date"]).reset_index(drop=True)

# Keep only necessary columns and remove zero/negative values
elasticity_data = df[["Product ID", "Price", "Units Sold"]].copy()
elasticity_data = elasticity_data.replace([np.inf, -np.inf], np.nan).dropna()

# Remove rows where price or units sold = 0 (log undefined)
elasticity_data = elasticity_data[(elasticity_data["Price"] > 0) & (elasticity_data["Units Sold"] > 0)]

# Create log variables
elasticity_data["log_price"] = np.log(elasticity_data["Price"])
elasticity_data["log_units"] = np.log(elasticity_data["Units Sold"])

# Prepare result container
results = []

# ------------------------------------------
# Compute elasticity per product
# ------------------------------------------
for pid, group in elasticity_data.groupby("Product ID"):
    if len(group) < 5:
        # Not enough data for regression
        elasticity = np.nan
    else:
        X = group["log_price"].values.reshape(-1, 1)
        y = group["log_units"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        elasticity = model.coef_[0][0]   # slope = elasticity

    results.append({
        "Product ID": pid,
        "price_elasticity": elasticity
    })

# Convert to DataFrame
elasticity_df = pd.DataFrame(results)

# ------------------------------------------
# CLASSIFY ELASTICITY
# ------------------------------------------
def classify(e):
    if pd.isna(e):
        return "Insufficient Data"
    if e < -1:
        return "High Elasticity"      # Demand very sensitive to price
    elif -1 <= e <= -0.5:
        return "Medium Elasticity"
    else:
        return "Low Elasticity"       # Demand not sensitive to price

elasticity_df["elasticity_class"] = elasticity_df["price_elasticity"].apply(classify)

# ------------------------------------------
# MERGE BACK TO ORIGINAL DATASET
# ------------------------------------------
final_df = df.merge(elasticity_df, on="Product ID", how="left")

# Save final dataset
final_df.to_csv("elasticity.csv", index=False)

print("Price elasticity and elasticity class created successfully!")
print(elasticity_df.head())
