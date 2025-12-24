import pandas as pd
import numpy as np

# ===================================
# LOAD DATASET
# ===================================
df = pd.read_csv("interaction.csv", low_memory=False)
print(f"📌 Loaded dataset → {df.shape}")

# ===================================
# REMOVE DUPLICATES
# ===================================
before = df.shape[0]
df = df.drop_duplicates()
print(f"✔ Removed duplicates → {before} → {df.shape[0]} rows")

# ===================================
# HANDLE MISSING VALUES
# ===================================
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(
    df[categorical_cols].mode().iloc[0]
)

print("✔ Missing values handled")

# ===================================
# REMOVE OUTLIERS (IQR METHOD)
# ===================================
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower) & (df_clean[col] <= upper)
        ]
    return df_clean

before_outliers = df.shape[0]
df = remove_outliers_iqr(df, numeric_cols)
print(f"✔ Outliers removed → {before_outliers} → {df.shape[0]} rows")

# ===================================
# SAVE CLEANED DATASET (NO SCALING)
# ===================================
df.to_csv("cleaned_features.csv", index=False)
print("🎉 Cleaning pipeline completed → saved as cleaned_features.csv")
