import pandas as pd
import numpy as np
import json
import os

# ============================================================
# CONFIG
# ============================================================

PARKS_FILE = "data/parques_coords.csv"
STREETS_FILE = "traffic_score_by_street.csv"
TRAFFIC_SCORES_FILE = "data/traffic_score_by_hour.csv"
MAIN_DATASET = "data/dataset_fluxos_hourly_2022.csv"
OUTPUT_FILE = "data/dataset_fluxos_hourly_2022_updated.csv"

# ============================================================
# 1. LOAD PARKS & EXTRACT COORDINATES
# ============================================================
print("[1/5] Loading parks data...", flush=True)
df_parks = pd.read_csv(PARKS_FILE)
print(f"   Loaded raw data. Rows: {len(df_parks)}", flush=True)

def extract_coords(pos_str):
    try:
        # It seems to be a string representation of a dict or JSON
        # Example: {"coordinates":[-9.122116,38.769492],"type":"Point"}
        # Note: GeoJSON usually uses [lon, lat]
        data = json.loads(pos_str)
        lon, lat = data["coordinates"]
        return lat, lon
    except Exception as e:
        print(f"Error parsing position: {pos_str} -> {e}")
        return np.nan, np.nan

df_parks["latitude"], df_parks["longitude"] = zip(*df_parks["position"].apply(extract_coords))
df_parks = df_parks[["id_parque", "latitude", "longitude"]].dropna()
print(f"   Loaded {len(df_parks)} parks with coordinates.")

# ============================================================
# 2. LOAD STREETS
# ============================================================
print("[2/5] Loading streets data...")
df_streets = pd.read_csv(STREETS_FILE)
# Get unique streets with their coordinates
unique_streets = df_streets.groupby("local")[["latitude", "longitude"]].mean().reset_index()
print(f"   Loaded {len(unique_streets)} unique streets.")

# ============================================================
# 3. FIND CLOSEST STREETS (1st and 2nd)
# ============================================================
print("[3/5] Mapping parks to closest streets...", flush=True)

# Calculate distance matrix (Parks x Streets)
# parks_coords: (N, 2), streets_coords: (M, 2)
parks_coords = df_parks[["latitude", "longitude"]].values
streets_coords = unique_streets[["latitude", "longitude"]].values
street_names = unique_streets["local"].values

# Euclidean distance matrix
# We use broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2) -> norm -> (N, M)
dists = np.sqrt(np.sum((parks_coords[:, np.newaxis, :] - streets_coords[np.newaxis, :, :]) ** 2, axis=2))

# Get indices of 1st and 2nd closest
# argsort sorts ascending, so we take first two columns
sorted_indices = np.argsort(dists, axis=1)
closest_indices = sorted_indices[:, 0]
second_closest_indices = sorted_indices[:, 1]

df_parks["closest_street"] = street_names[closest_indices]
df_parks["second_closest_street"] = street_names[second_closest_indices]

print("   Park -> Closest Streets Mapping (Sample):")
print(df_parks[["id_parque", "closest_street", "second_closest_street"]].head())

# ============================================================
# 4. PREPARE TRAFFIC DATA LOOKUPS
# ============================================================
print("[4/5] Preparing traffic data lookups...", flush=True)
df_main = pd.read_csv(MAIN_DATASET)
df_traffic = pd.read_csv(TRAFFIC_SCORES_FILE)

# Ensure date columns match
df_main["date"] = pd.to_datetime(df_main["date"])
df_traffic["date"] = pd.to_datetime(df_traffic["date"])

# Filter invalid scores for averages (exclude <= 0 and NaN)
valid_traffic = df_traffic[(df_traffic["score"] > 0) & (df_traffic["score"].notna())].copy()
valid_traffic["month"] = valid_traffic["date"].dt.month

# A. Monthly Average per Street
monthly_avg = valid_traffic.groupby(["local", "month"])["score"].mean().reset_index()
monthly_avg.rename(columns={"score": "monthly_avg_score"}, inplace=True)

# B. Random Street Score per (Date, Hour)
# We shuffle and take the first one for each group to simulate randomness
random_traffic = valid_traffic.sample(frac=1, random_state=42).groupby(["date", "hour"]).first().reset_index()
random_traffic = random_traffic[["date", "hour", "score"]].rename(columns={"score": "random_score"})

# ============================================================
# 5. MERGE AND UPDATE WITH FALLBACKS
# ============================================================
print("[5/5] Merging and updating traffic scores...", flush=True)

# 1. Merge Park Info (Closest Streets)
df_merged = df_main.merge(
    df_parks[["id_parque", "closest_street", "second_closest_street"]], 
    left_on="parking_lot", 
    right_on="id_parque", 
    how="left"
)

# 2. Merge 1st Closest Score
df_merged = df_merged.merge(
    df_traffic[["local", "date", "hour", "score"]],
    left_on=["closest_street", "date", "hour"],
    right_on=["local", "date", "hour"],
    how="left"
)
df_merged.rename(columns={"score": "score_1st"}, inplace=True)
df_merged.drop(columns=["local"], inplace=True)

# 3. Merge 2nd Closest Score
df_merged = df_merged.merge(
    df_traffic[["local", "date", "hour", "score"]],
    left_on=["second_closest_street", "date", "hour"],
    right_on=["local", "date", "hour"],
    how="left"
)
df_merged.rename(columns={"score": "score_2nd"}, inplace=True)
df_merged.drop(columns=["local"], inplace=True)

# 4. Merge Random Score
df_merged = df_merged.merge(
    random_traffic,
    on=["date", "hour"],
    how="left"
)

# 5. Merge Monthly Average (of Closest Street)
df_merged["month"] = df_merged["date"].dt.month
df_merged = df_merged.merge(
    monthly_avg,
    left_on=["closest_street", "month"],
    right_on=["local", "month"],
    how="left"
)
df_merged.drop(columns=["local"], inplace=True)

# APPLY FALLBACK LOGIC
# Order: Existing -> 1st -> 2nd -> Random -> Monthly Avg
print("   Applying fallback logic...", flush=True)

# Initialize with existing, fill with 1st
df_merged["final_score"] = df_merged["traffic_score"].combine_first(df_merged["score_1st"])

# Fill with 2nd
df_merged["final_score"] = df_merged["final_score"].combine_first(df_merged["score_2nd"])

# Fill with Random
df_merged["final_score"] = df_merged["final_score"].combine_first(df_merged["random_score"])

# Fill with Monthly Avg
df_merged["final_score"] = df_merged["final_score"].combine_first(df_merged["monthly_avg_score"])

# Update column
df_merged["traffic_score"] = df_merged["final_score"]

# Statistics
total_rows = len(df_merged)
missing_after = df_merged["traffic_score"].isna().sum()
print(f"   Total Rows: {total_rows}")
print(f"   Missing After Update: {missing_after} ({missing_after/total_rows:.2%})")

# Drop auxiliary columns
cols_to_drop = [
    "id_parque", "closest_street", "second_closest_street", 
    "score_1st", "score_2nd", "random_score", "monthly_avg_score", 
    "month", "final_score"
]
df_final = df_merged.drop(columns=cols_to_drop, errors="ignore")

# Save
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"[DONE] Saved updated dataset to: {OUTPUT_FILE}")
