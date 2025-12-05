import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("daily_fixed_dataset.csv")
print("\n Loaded file → daily_fixed_dataset.csv\n")

# Detect categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Detected Categorical Columns:", cat_cols)

# ----------------------------
# Encoding rules
# ----------------------------
LOW_CARDINALITY_LIMIT = 20  # threshold

le = LabelEncoder()

for col in cat_cols:

    unique_count = df[col].nunique()

    # ----------------------------
    # Low cardinality → Label Encoding
    # ----------------------------
    if unique_count <= LOW_CARDINALITY_LIMIT:
        print(f"Label Encoding → {col}")
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    # ----------------------------
    # High cardinality → Frequency Encoding
    # ----------------------------
    else:
        print(f"Frequency Encoding → {col}")
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)

# ----------------------------
# Save encoded dataset
# ----------------------------
df.to_csv("encoded_dataset.csv", index=False)
print("\n🎉 Encoding Completed → encoded_dataset.csv")
print(df.head())
