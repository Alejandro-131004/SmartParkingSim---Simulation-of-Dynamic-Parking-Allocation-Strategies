import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================

PARKS = ["P023", "P024", "P040", "P064"]
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

# VERY IMPORTANT: treat empty string "" as missing value
df["traffic_score"] = df["traffic_score"].replace("", np.nan)

print("[OK] Dataset loaded.")

# Generate full date range for 2022 from Jan 1 to Jun 30
full_dates = pd.date_range("2022-01-01", "2022-06-30", freq="D")

# ============================================================
# PROCESS PER PARK
# ============================================================

print("[2/3] Computing missing traffic per park...")

for park in PARKS:
    print(f"   → Processing {park}...")

    dfp = df[df["parking_lot"] == park].copy()

    # count missing values per date
    missing = (
        dfp[dfp["traffic_score"].isna()]
        .groupby(dfp["date"].dt.date)
        .size()
        .reindex(full_dates.date, fill_value=0)  # force 1 Jan to appear even if 0
    )

    out = pd.DataFrame({
        "date": missing.index,
        "missing_rows": missing.values
    })

    output_path = os.path.join(OUTPUT_FOLDER, f"missing_traffic_{park}.csv")
    out.to_csv(output_path, index=False)

    print(f"      Saved: {output_path}")

print("[3/3] DONE.")
