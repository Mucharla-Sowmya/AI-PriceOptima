import pandas as pd

# Load datasets
sales = pd.read_csv("sales_dataset.csv")
inv = pd.read_csv("inventory_dataset.csv")   # <-- use modified one

# Select mandatory sales fields
sales_req = sales[[
    "Date", "Product ID", "Units Sold", "Price", "Revenue", "Store ID"
]]

# Select inventory fields + newly added ones
inv_req = inv[[
    "Product ID",
    "Inventory Level",
    "Restock_Date",
    "Store ID",
    "AvgPrice",     # NEW
    "Cost"          # NEW
]]

# Rename inventory columns
inv_req = inv_req.rename(columns={
    "Inventory Level": "Stock Level",
    "Store ID": "Warehouse/Store ID"
})

# Safe merge: Product ID + Store ID
combined = pd.merge(
    sales_req,
    inv_req,
    left_on=["Product ID", "Store ID"],
    right_on=["Product ID", "Warehouse/Store ID"],
    how="left"
)

# Save combined dataset
combined.to_csv("combined_mandatory_fields_dataset.csv", index=False)

print("✅ MERGE DONE — Combined dataset created successfully!")
