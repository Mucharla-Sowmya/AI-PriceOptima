import pandas as pd

# Load the dataset
df = pd.read_csv("../milestone-1/combined_dataset.csv")

# Safe robust datetime parsing
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Restock_Date"] = pd.to_datetime(df["Restock_Date"], dayfirst=True, errors="coerce")

# Sort by Product & Date (recommended for time-series features)
df = df.sort_values(by=["Product ID", "Date"]).reset_index(drop=True)

# ------------------------
# TIME-BASED FEATURES
# ------------------------

df["day"] = df["Date"].dt.day.astype("Int64")
df["month"] = df["Date"].dt.month.astype("Int64")
df["year"] = df["Date"].dt.year.astype("Int64")

df["day_of_week"] = df["Date"].dt.dayofweek.astype("Int64")
df["day_name"] = df["Date"].dt.day_name()

df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("Int64")

# ------------------------
# SEASON FEATURE (India climate pattern)
# ------------------------

def get_season(month):
    if pd.isna(month):
        return None
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    elif month in [10, 11]:
        return "post_monsoon"

df["season"] = df["month"].apply(get_season)

# ------------------------
# HOLIDAY FEATURES
# ------------------------

holiday_dict = {
    "01-01-2022": "New Year",
    "14-01-2022": "Pongal",
    "26-01-2022": "Republic Day",
    "18-03-2022": "Holi",
    "02-05-2022": "Eid",
    "15-08-2022": "Independence Day",
    "24-10-2022": "Diwali",
    "25-12-2022": "Christmas",
}

holiday_map = {
    pd.to_datetime(k, dayfirst=True): v for k, v in holiday_dict.items()
}

df["is_holiday"] = df["Date"].apply(lambda x: 1 if x in holiday_map else 0).astype("Int64")
df["holiday_name"] = df["Date"].apply(lambda x: holiday_map.get(x, ""))

# ------------------------
# EXTRA DATE FEATURES (NA-safe)
# ------------------------

df["week_of_year"] = df["Date"].dt.isocalendar().week.astype("Int64")
df["quarter"] = df["Date"].dt.quarter.astype("Int64")
df["day_of_year"] = df["Date"].dt.dayofyear.astype("Int64")
df["is_month_start"] = df["Date"].dt.is_month_start.astype("Int64")
df["is_month_end"] = df["Date"].dt.is_month_end.astype("Int64")

# ------------------------
# SAVE FINAL FEATURES
# ------------------------

df.to_csv("time.csv", index=False)

print("Time-based features created successfully → time.csv")
print(df.head())
