import pandas as pd
import numpy as np
import yaml
import os

# ============================================================
# CONFIG
# ============================================================

PARK_ID = "P023"
RAW_DATA_PATH = "data/dataset_fluxos_hourly_2022.csv"
TRAFFIC_PATH = "data/traffic_score_by_hour.csv"

# Updated to match the path defined in runner.py / parking_P023.yaml
OUTPUT_CSV = "data/P023_sim/dataset_P023_simready_updated.csv"

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("config", exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("[1/6] Loading dataset...", flush=True)

df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

df["date"] = pd.to_datetime(df["date"])
df_p23 = df[df["parking_lot"] == PARK_ID].copy()
df_p23 = df_p23.sort_values(["date", "hour"])

# DROP legacy columns (entries/exits) if they exist
cols_to_drop = ["entries", "exits", "taxa_fluxo_hora", "fluxo_total"]
df_p23 = df_p23.drop(columns=[c for c in cols_to_drop if c in df_p23.columns], errors="ignore")

print(f"[OK] Loaded {len(df_p23)} rows for {PARK_ID}. (Occupancy Only)")

# ============================================================
# 2. COMPUTE HOURLY MEANS (for heavy-missing days)
# ============================================================

print("[2/6] Computing hourly means...", flush=True)

hourly_means = df_p23.groupby("hour")["occupancy"].mean().round().astype(int).to_dict()

print("[OK] Hourly means computed.")

# ============================================================
# 3. SAFE HOURLY REINDEX + IMPUTATION
# ============================================================

print("[3/6] Imputing missing hours safely...", flush=True)

all_days = sorted(df_p23["date"].dt.date.unique())
final_rows = []

for day in all_days:
    # Filter for the specific day
    daily = df_p23[df_p23["date"].dt.date == day].copy()
    
    # Reindex to ensure all 24 hours exist (0-23)
    daily = daily.set_index("hour").reindex(range(24))
    daily["date"] = pd.to_datetime(day)

    # Only impute Occupancy
    col = "occupancy"
    missing = daily[col].isna().sum()

    if missing <= 4:
        # Short gaps: Forward/Backward fill
        daily[col] = daily[col].ffill().bfill()
    else:
        # Long gaps: Use historical hourly mean
        for h in range(24):
            if pd.isna(daily.loc[h, col]):
                daily.loc[h, col] = hourly_means.get(h, 0)

    # Convert back to int after imputation
    daily["occupancy"] = daily["occupancy"].round().astype(int)

    final_rows.append(daily.reset_index())

df_clean = pd.concat(final_rows).sort_values(["date", "hour"])
print("[OK] Dataset cleaned correctly.")

# ==========================================
# 4 — Load traffic-score data
# ==========================================
print("[4/6] Loading traffic score data...", flush=True)

traffic = pd.read_csv(TRAFFIC_PATH)
traffic.columns = traffic.columns.str.lower().str.strip()
traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce")

# Only keep 2022
traffic_2022 = traffic[traffic["date"].dt.year == 2022].copy()

# Normalize local names
traffic_2022["local"] = (
    traffic_2022["local"]
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
    .str.strip()
)

TARGET_LOCAL = "Avenida da Republica"
traffic_2022 = traffic_2022[traffic_2022["local"] == TARGET_LOCAL].copy()

print(f"[OK] traffic_score_by_hour has {len(traffic_2022)} rows for 2022.")

# Rename score → traffic_score
traffic_2022 = traffic_2022.rename(columns={"score": "traffic_score"})
traffic_2022 = traffic_2022[["date", "hour", "traffic_score"]]

# Ensure hour types match
traffic_2022["hour"] = traffic_2022["hour"].astype(int)
df_clean["hour"] = df_clean["hour"].astype(int)

# ==========================================
# 5 — Merge traffic score into df_clean
# ==========================================
print("[5/6] Merging traffic data...", flush=True)

df_final = df_clean.merge(
    traffic_2022,
    how="left",
    on=["date", "hour"]
)

# ==========================================
# 6 — Fix column names & Cleanup
# ==========================================
if "traffic_score_x" in df_final.columns:
    df_final["traffic_score"] = df_final["traffic_score_x"]

if "traffic_score_y" in df_final.columns:
    df_final["traffic_score"] = df_final["traffic_score_y"]

df_final = df_final.drop(
    columns=[c for c in df_final.columns if "_x" in c or "_y" in c],
    errors="ignore"
)

# Ensure strictly no legacy columns remain
df_final = df_final.drop(columns=["entries", "exits"], errors="ignore")

# ==========================================
# 7 — Validate & Save
# ==========================================
missing_traffic = df_final["traffic_score"].isna().sum()
print(f"[OK] Missing traffic_score after merge: {missing_traffic}")

print(f"[SAVE] Saving simulation-ready dataset to: {OUTPUT_CSV}")
df_final.to_csv(OUTPUT_CSV, index=False)
print("[DONE] Dataset generation complete.")