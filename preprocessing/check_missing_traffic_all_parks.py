import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataset_fluxos_hourly_2022.csv"
OUTPUT_FOLDER = "data/missing_traffic"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("[1/3] Loading dataset...")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])

# Ensure we have a datetime column combining date and hour for proper indexing
# The dataset has 'date' (YYYY-MM-DD) and 'hour' (0-23)
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str) + ":00:00")

# VERY IMPORTANT: treat empty string "" as missing value
if df["traffic_score"].dtype == object:
    df["traffic_score"] = df["traffic_score"].replace("", np.nan)

# Get all unique parks from the dataset
PARKS = df["parking_lot"].unique()
PARKS.sort()

print(f"[OK] Dataset loaded. Found {len(PARKS)} parks: {', '.join(PARKS)}")

# Generate full hourly range for 2022
# We'll use the min and max date from the dataset, or force full 2022 if preferred.
# Let's use full year 2022 as per context of "2022 dataset"
full_range = pd.date_range("2022-01-01 00:00:00", "2022-12-31 23:00:00", freq="H")

# ============================================================
# PROCESS PER PARK
# ============================================================

print("[2/3] Computing missing traffic per park...")

for park in PARKS:
    print(f"   → Processing {park}...")

    dfp = df[df["parking_lot"] == park].copy()
    
    # Set index to datetime to allow reindexing
    dfp = dfp.set_index("datetime")
    
    # Reindex to full hourly range
    # This will create rows with NaN for missing hours
    dfp_reindexed = dfp.reindex(full_range)
    
    # 1. Missing Rows: The index didn't exist in original data (all cols are NaN after reindex)
    # We can check if 'parking_lot' is NaN, because it should be present if the row existed
    is_missing_row = dfp_reindexed["parking_lot"].isna()
    
    # 2. NaN Traffic: The row exists, but traffic_score is NaN
    # We check where row is NOT missing, but traffic_score IS missing
    is_nan_traffic = (~is_missing_row) & (dfp_reindexed["traffic_score"].isna())
    
    # Create a summary dataframe
    summary = pd.DataFrame({
        "missing_rows": is_missing_row,
        "nan_traffic": is_nan_traffic
    })
    
    # Group by date (day) to get daily counts
    daily_stats = summary.groupby(summary.index.date).sum()
    daily_stats.index.name = "date"
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_FOLDER, f"missing_traffic_{park}.csv")
    daily_stats.to_csv(output_path)

    print(f"      Saved: {output_path}")

print("[3/3] DONE.")
