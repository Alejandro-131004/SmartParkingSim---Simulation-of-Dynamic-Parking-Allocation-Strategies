import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH = "data/dataset_P023_simready.csv"
TRAFFIC_PATH = "data/traffic_score_by_hour.csv"

# ============================================================
# LOAD DATA
# ============================================================
print("[1/4] Loading data...")

df = pd.read_csv(DATASET_PATH)
traffic = pd.read_csv(TRAFFIC_PATH)

# Normalizar colunas
df.columns = df.columns.str.lower()
traffic.columns = traffic.columns.str.lower()

df["date"] = pd.to_datetime(df["date"])
traffic["date"] = pd.to_datetime(traffic["date"])

print("[OK] Loaded dataset and traffic_score_by_hour.")


# ============================================================
# IDENTIFY MISSING TRAFFIC SCORE IN FINAL DATASET
# ============================================================
print("\n[2/4] Detecting missing traffic scores...")

missing = df[df["traffic_score"].isna()]

print(f"[INFO] Missing traffic_score rows: {len(missing)}")

# Missing por dia
missing_by_day = (
    missing.groupby(missing["date"].dt.date)
    .size()
    .reset_index(name="missing_rows")
)

# Missing por hora (global)
missing_by_hour = (
    missing.groupby("hour")
    .size()
    .reset_index(name="missing_count")
)

# Mostrar resumo
print("\n===== Missing Traffic Score by Day =====")
print(missing_by_day.to_string(index=False))

print("\n===== Missing Traffic Score by Hour =====")
print(missing_by_hour.to_string(index=False))


# ============================================================
# CHECK IF THOSE DAYS EXIST IN traffic_score_by_hour CSV
# ============================================================
print("\n[3/4] Checking traffic_score_by_hour availability...")

available_days = traffic["date"].dt.date.unique()
missing["day"] = missing["date"].dt.date

missing["exists_in_traffic_file"] = missing["day"].isin(available_days)

missing_check = (
    missing.groupby(["day", "exists_in_traffic_file"])
    .size()
    .reset_index(name="rows")
)

print("\n===== Missing Days vs Traffic File Availability =====")
print(missing_check.to_string(index=False))


# Dias completamente ausentes do ficheiro de tráfego
missing_days_no_data = (
    missing_check[missing_check["exists_in_traffic_file"] == False]["day"].unique()
)

print("\n===== Days with NO TRAFFIC DATA AT ALL =====")
for d in missing_days_no_data:
    print(d)


# ============================================================
# EXPORT OPTIONAL REPORT
# ============================================================
print("\n[4/4] Saving report to data/missing_traffic_report.csv...")

missing_report = missing_by_day.copy()
missing_report.to_csv("data/missing_traffic_report.csv", index=False)

print("[OK] Report saved: data/missing_traffic_report.csv")
