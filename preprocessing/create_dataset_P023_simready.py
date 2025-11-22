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

OUTPUT_CSV = "data/dataset_P023_simready.csv"
OUTPUT_YAML = "config/parking_P023.yaml"

os.makedirs("data", exist_ok=True)
os.makedirs("config", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("[1/6] Loading dataset...")

df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

df["date"] = pd.to_datetime(df["date"])
df_p23 = df[df["parking_lot"] == PARK_ID].copy()
df_p23 = df_p23.sort_values(["date", "hour"])

print(f"[OK] Loaded {len(df_p23)} rows for {PARK_ID}.")

# ============================================================
# 2. COMPUTE HOURLY MEANS (for heavy-missing days)
# ============================================================

print("[2/6] Computing hourly means...")

hourly_means = df_p23.groupby("hour")["occupancy"].mean().round().astype(int).to_dict()

print("[OK] Hourly means computed.")

# ============================================================
# 3. SAFE HOURLY REINDEX + IMPUTATION
# ============================================================

print("[3/6] Imputing missing hours safely...")

all_days = sorted(df_p23["date"].dt.date.unique())
final_rows = []

for day in all_days:
    daily = df_p23[df_p23["date"].dt.date == day].copy()
    daily = daily.set_index("hour").reindex(range(24))

    daily["date"] = pd.to_datetime(day)

    for col in ["entries", "exits", "occupancy"]:
        if col not in daily.columns:
            continue

        missing = daily[col].isna().sum()

        if missing <= 4:
            daily[col] = daily[col].ffill().bfill()
        else:
            for h in range(24):
                if pd.isna(daily.loc[h, col]):
                    daily.loc[h, col] = hourly_means[h]

    # convert back to int after imputation
    daily["occupancy"] = daily["occupancy"].round().astype(int)

    final_rows.append(daily.reset_index())

df_clean = pd.concat(final_rows).sort_values(["date", "hour"])
print("[OK] Dataset cleaned correctly.")

# ==========================================
# 4 — Load traffic-score data
# ==========================================
print("[4/6] Loading traffic score data...")

traffic_path = "data/traffic_score_by_hour.csv"
traffic = pd.read_csv(traffic_path)

# Normalize column names
traffic.columns = traffic.columns.str.lower().str.strip()

# Convert date
traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce")

# Only keep 2022
traffic_2022 = traffic[traffic["date"].dt.year == 2022].copy()

# Normalize local names to remove accents
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

# Reduce to needed columns
traffic_2022 = traffic_2022[["date", "hour", "traffic_score"]]

# Ensure hour types match
traffic_2022["hour"] = traffic_2022["hour"].astype(int)
df_clean["hour"] = df_clean["hour"].astype(int)

# ==========================================
# 5 — Merge traffic score into df_clean
# ==========================================
df_final = df_clean.merge(
    traffic_2022,
    how="left",
    on=["date", "hour"]
)

# ==========================================
# 6 — Fix column names if needed
# ==========================================
if "traffic_score_x" in df_final.columns:
    df_final["traffic_score"] = df_final["traffic_score_x"]

if "traffic_score_y" in df_final.columns:
    df_final["traffic_score"] = df_final["traffic_score_y"]

# Remove leftovers (_x / _y)
df_final = df_final.drop(
    columns=[c for c in df_final.columns if "_x" in c or "_y" in c],
    errors="ignore"
)

# ==========================================
# 7 — Validate traffic_score
# ==========================================
missing = df_final["traffic_score"].isna().sum()
print(f"[OK] Missing traffic_score after merge: {missing}")
