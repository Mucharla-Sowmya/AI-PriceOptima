# ============================================================
# Milestone 2 – Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("../milestone-1/combined_dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 2. Create Profit Column
# ------------------------------------------------------------
df["Profit"] = df["Revenue"] - (df["Units Sold"] * df["Cost"])

# ------------------------------------------------------------
# 3. Select Required Columns
# ------------------------------------------------------------
eda_cols = [
    "Price",
    "Units Sold",
    "Revenue",
    "Profit",
    "Stock Level"
]

eda_df = df[eda_cols]

# ------------------------------------------------------------
# 4. Missing Values Detection (PRINT ONLY)
# ------------------------------------------------------------
print("\nMissing Values:")
print(eda_df.isnull().sum())

# ------------------------------------------------------------
# 5. Basic Statistical Analysis (PRINT ONLY)
# ------------------------------------------------------------
print("\nStatistical Summary:")
print(eda_df.describe())

# ------------------------------------------------------------
# 6. Relationship Insights (PRINT CORRELATION VALUES)
# ------------------------------------------------------------
print("\nCorrelation Values:")
print(eda_df.corr())

# ------------------------------------------------------------
# 7. SINGLE REQUIRED VISUALIZATION (HEATMAP)
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    eda_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap: Price, Demand & Inventory")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Outlier Detection (PRINT ONLY – IQR METHOD)
# ------------------------------------------------------------
print("\nOutlier Detection (IQR Method):")

for col in eda_cols:
    Q1 = eda_df[col].quantile(0.25)
    Q3 = eda_df[col].quantile(0.75)
    IQR = Q3 - Q1

    outliers = eda_df[
        (eda_df[col] < Q1 - 1.5 * IQR) |
        (eda_df[col] > Q3 + 1.5 * IQR)
    ]

    print(f"{col}: {outliers.shape[0]} outliers")

print("\nEDA Completed Successfully!")
