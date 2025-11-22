import pandas as pd

# === Load datasets ===
df_fluxos = pd.read_csv("data/dataset_fluxos_hourly_2022.csv", low_memory=False)
df_temp = pd.read_csv("data/temp_min_max_station1.csv")

# === Prepare date columns ===
df_fluxos["data"] = pd.to_datetime(df_fluxos["data"]).dt.date
df_temp["dia"] = pd.to_datetime(df_temp["dia"]).dt.date

# === Merge by date ===
merged = df_fluxos.merge(df_temp, left_on="data", right_on="dia", how="left")
if "dia_y" in merged.columns:
    merged = merged.drop(columns=["dia_y"], errors="ignore")

# === Translate column names to English ===
rename_map = {
    "parque": "parking_lot",
    "capacidade": "capacity",
    "dia": "day_of_year",
    "data": "date",
    "hora": "hour",
    "dia_semana": "weekday",
    "entradas": "entries",
    "saidas": "exits",
    "ocupacao": "occupancy",
    "traffic_score": "traffic_score",
    "fluxo_total": "total_flow",
    "taxa_fluxo_hora": "hourly_flow_rate",
    "temp_min": "temp_min_c",
    "temp_max": "temp_max_c",
}

merged = merged.rename(columns=rename_map)

# === Save output ===
output_path = "dataset_fluxos_hourly_2022.csv"
merged.to_csv(output_path, index=False)

print(f"Saved enriched dataset with weather and English column names: {output_path}")
print(f"Columns: {list(merged.columns)}")
