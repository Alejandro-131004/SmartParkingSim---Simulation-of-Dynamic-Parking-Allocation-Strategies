import pandas as pd
import numpy as np
import yaml
import os

# ============================================================
# CONFIG
# ============================================================

PARK_ID = "P023"
# Pointing to the output of our previous integrated script
RAW_DATA_PATH = "parking_occupancy_traffic_2022.csv" 
OUTPUT_CSV = "data/dataset_P023_simready.csv"
OUTPUT_YAML = "config/parking_P023.yaml"

os.makedirs("data", exist_ok=True)
os.makedirs("config", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("[1/6] Loading integrated dataset...")

df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

# Aligning column names from integrated script to this script's logic
# Integrated: 'data', 'hora', 'parque', 'ocupacao'
df = df.rename(columns={
    'data': 'date', 
    'hora': 'hour', 
    'parque': 'parking_lot', 
    'ocupacao': 'occupancy'
})

df["date"] = pd.to_datetime(df["date"])
df_p23 = df[df["parking_lot"] == PARK_ID].copy()
df_p23 = df_p23.sort_values(["date", "hour"])

print(f"[OK] Loaded {len(df_p23)} rows for {PARK_ID}.")

# ============================================================
# 2. COMPUTE HOURLY MEANS
# ============================================================

print("[2/6] Computing hourly means for imputation...")
# This creates a "typical day" profile to fill long gaps
hourly_means = df_p23.groupby("hour")["occupancy"].mean().round().astype(int).to_dict()

# ============================================================
# 3. SAFE HOURLY REINDEX + IMPUTATION
# ============================================================

print("[3/6] Imputing missing hours...")

all_days = sorted(df_p23["date"].dt.date.unique())
final_rows = []

for day in all_days:
    daily = df_p23[df_p23["date"].dt.date == day].copy()
    
    # Ensure all 24 hours exist for every day
    daily = daily.set_index("hour").reindex(range(24))
    daily["date"] = pd.to_datetime(day)
    daily["parking_lot"] = PARK_ID

    # Fill 'occupancy' (and 'traffic_score' if available)
    # We check if gap is small (<= 4h) use interpolation, else use hourly mean
    missing_count = daily["occupancy"].isna().sum()
    
    if missing_count > 0:
        if missing_count <= 4:
            daily["occupancy"] = daily["occupancy"].ffill().bfill()
        else:
            for h in range(24):
                if pd.isna(daily.loc[h, "occupancy"]):
                    daily.loc[h, "occupancy"] = hourly_means.get(h, 0)

    # Convert to int after imputation
    daily["occupancy"] = daily["occupancy"].round().astype(int)
    final_rows.append(daily.reset_index())

df_clean = pd.concat(final_rows).sort_values(["date", "hour"])
print("[OK] Dataset timeline filled and imputed.")

# ==========================================
# 4 — Export Simulation Config (YAML)
# ==========================================
# Adding the YAML export logic which is helpful for Simulators
print("[4/6] Exporting configuration...")

# Get capacity from original data if available, else max of occupancy
capacity = int(df_p23['capacidade'].iloc[0]) if 'capacidade' in df_p23.columns else int(df_clean['occupancy'].max())

config = {
    'park_id': PARK_ID,
    'capacity': capacity,
    'lat': float(df_p23['lat'].iloc[0]) if 'lat' in df_p23.columns else 0.0,
    'lon': float(df_p23['lon'].iloc[0]) if 'lon' in df_p23.columns else 0.0,
    'total_steps': len(df_clean)
}

with open(OUTPUT_YAML, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# ==========================================
# 5 — Save final CSV
# ==========================================
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"[OK] Simulation-ready file saved: {OUTPUT_CSV}")
print(f"[OK] Configuration saved: {OUTPUT_YAML}")