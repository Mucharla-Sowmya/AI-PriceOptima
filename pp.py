#   1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

#2. LOAD THE DATASET
file_path = "retail_store_inventory.csv" 
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully!\n")

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

#3. HANDLE MISSING VALUES

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Filling numeric columns with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Filling categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#4. REMOVE DUPLICATES
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"\nDuplicates Removed: {before - after}")

#5. CORRECT DATA TYPES
print("\nData Types Before Conversion:")
print(df.dtypes)
    
# Example: ensure product_id is string
if 'product_id' in df.columns:
    df['product_id'] = df['product_id'].astype(str)

print("\nData Types After Conversion:")
print(df.dtypes)

#6. OUTLIER TREATMENT

def treat_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    dataframe[column] = np.where(
        dataframe[column] < lower, lower,
        np.where(dataframe[column] > upper, upper, dataframe[column])
    )

# Apply to key numerical columns if present
for col in ['price', 'units_sold', 'inventory_available', 'competitor_price']:
    if col in df.columns:
        treat_outliers_iqr(df, col)

print("\nOutlier Treatment Completed!")

#8. FINAL CLEANED DATASET

print("\nFinal Clean Dataset Shape:", df.shape)
print("\nSummary:")
print(df.describe(include='all'))

# Save for next steps
df.to_csv("cleaned_retail_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_retail_dataset.csv")
