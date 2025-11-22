import pandas as pd
import numpy as np
import time
from math import radians, sin, cos, sqrt, atan2

# ---------------------------------------------------------
# Helper: timer checkpoint
# ---------------------------------------------------------
def checkpoint(msg, start):
    now = time.time()
    print(f"[OK] {msg} — {now - start:.2f}s")
    return now

print("\n==============================================")
print("      PARKING SELECTION – PROGRESS TRACKER")
print("==============================================\n")

# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

t = time.time()
print("[1/10] A carregar dataset principal...")
df = pd.read_csv("data/dataset_fluxos_hourly_2022.csv")
t = checkpoint("data/dataset_fluxos_hourly_2022 carregado", t)

print("[2/10] A carregar Excel dos parques (com coordenadas)...")
parks_raw = pd.read_excel("parques-estacionamento-1s-2022.xlsx")
t = checkpoint("Excel dos parques carregado", t)

# ---------------------------------------------------------
# 2. Normalize columns + extract coordinates from `position`
# ---------------------------------------------------------

print("[3/10] A normalizar colunas e extrair coordenadas...")
df.columns = df.columns.str.strip().str.lower()
parks_raw.columns = parks_raw.columns.str.strip().str.lower()

def extract_coords(position_str):
    try:
        if isinstance(position_str, str):
            d = eval(position_str)
            if "coordinates" in d:
                lon, lat = d["coordinates"]
                return lat, lon
    except:
        pass
    return np.nan, np.nan

parks_raw["lat"], parks_raw["lon"] = zip(*parks_raw["position"].apply(extract_coords))

coords_df = (
    parks_raw[["id_parque", "lat", "lon"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={"id_parque": "parking_lot"})
)

t = checkpoint("Coordenadas extraídas", t)

# ---------------------------------------------------------
# 3. Detect invalid parks (negative occupancy or frozen >20h)
# ---------------------------------------------------------

print("[4/10] A identificar parques inválidos (valores negativos e períodos constantes)...")

df = df.sort_values(["parking_lot", "date", "hour"]).reset_index(drop=True)

# 3.1 parks with occupancy < 0
bad_neg = df[df["occupancy"] < 0]["parking_lot"].unique().tolist()

# 3.2 parks with >20h constant occupancy
def has_long_constant_period(group, max_hours_same=20):
    group = group.copy()
    group["change"] = group["occupancy"].diff().ne(0).astype(int)
    group["segment_id"] = group["change"].cumsum()
    return group["segment_id"].value_counts().max() > max_hours_same

bad_constant = []
for lot, g in df.groupby("parking_lot"):
    if has_long_constant_period(g):
        bad_constant.append(lot)

bad_parks = set(bad_neg) | set(bad_constant)

print(f"   -> Parques com occupancy < 0: {len(bad_neg)}")
print(f"   -> Parques com >20h constantes: {len(bad_constant)}")
print(f"   -> TOTAL parques inválidos removidos: {len(bad_parks)}")

t = checkpoint("Parques inválidos identificados", t)

# ---------------------------------------------------------
# 4. Filter & compute variance
# ---------------------------------------------------------

print("[5/10] A filtrar parques válidos e calcular variâncias...")

df_valid = df[~df["parking_lot"].isin(bad_parks)].copy()
df_valid["date"] = pd.to_datetime(df_valid["date"]).dt.date

variance_df = (
    df_valid.groupby("parking_lot")["occupancy"]
    .var()
    .reset_index(name="variance")
)

mean_df = (
    df_valid.groupby("parking_lot")["occupancy"]
    .mean()
    .reset_index(name="mean_occupancy")
)

variance_df = variance_df.merge(mean_df, on="parking_lot")
variance_df["variance_percent"] = (
    variance_df["variance"] / variance_df["mean_occupancy"] * 100
)

t = checkpoint("Variâncias calculadas", t)

# ---------------------------------------------------------
# 5. EXTRA STEP — missing hours per park + missing_days
# ---------------------------------------------------------

print("[6/10] A verificar horas faltantes por parque...")

expected_hours = set(range(24))
missing_info = {}
missing_days_count = {}

for lot, glot in df_valid.groupby("parking_lot"):
    lot_missing = {}

    for day, gday in glot.groupby("date"):
        hours = set(gday["hour"].unique())
        missing = sorted(list(expected_hours - hours))
        if missing:
            lot_missing[str(day)] = missing

    missing_info[lot] = lot_missing
    missing_days_count[lot] = len(lot_missing)

def encode_missing(d):
    return "{}" if not d else str(d)

missing_df = pd.DataFrame({
    "parking_lot": list(missing_info.keys()),
    "missing_hours_by_day": [encode_missing(missing_info[lot]) for lot in missing_info],
    "missing_days": [missing_days_count[lot] for lot in missing_info]
})

t = checkpoint("Horas faltantes identificadas", t)

# ---------------------------------------------------------
# 6. Add capacity
# ---------------------------------------------------------

print("[7/10] A adicionar capacidade dos parques...")

capacity_map = {
    "P064": 248,
    "P040": 118,
    "P023": 180,
    "P024": 76
}

capacity_df = pd.DataFrame({
    "parking_lot": list(capacity_map.keys()),
    "capacity": list(capacity_map.values())
})

t = checkpoint("Capacidades adicionadas", t)

# ---------------------------------------------------------
# 7. Merge everything (coords + missing hours + capacity)
# ---------------------------------------------------------

print("[8/10] A juntar coordenadas, missing_hours e capacidade...")

final_df = variance_df.merge(coords_df, on="parking_lot", how="left")
final_df = final_df.merge(missing_df, on="parking_lot", how="left")
final_df = final_df.merge(capacity_df, on="parking_lot", how="left")

t = checkpoint("Dados combinados", t)

# ---------------------------------------------------------
# 8. FIX lat/lon types (critical fix)
# ---------------------------------------------------------

print("[8.1] A converter lat/lon para float...")

final_df["lat"] = pd.to_numeric(final_df["lat"], errors="coerce")
final_df["lon"] = pd.to_numeric(final_df["lon"], errors="coerce")

print(final_df[["parking_lot", "lat", "lon"]])  # debug opcional

t = checkpoint("lat/lon convertidos", t)

# ---------------------------------------------------------
# 9. Distance to interest points
# ---------------------------------------------------------

print("[9/10] A calcular distâncias corretas...")

interest_points = {
    "Avenida da Republica": (38.7485, -9.1487),
    "Avenida das Forças Armadas": (38.7394, -9.1525),
    "Avenida de Berlim": (38.7672, -9.1032),
    "Calçada de Carriche": (38.7707, -9.1529),
    "Marquês de Pombal": (38.7253, -9.1502),
    "Restauradores": (38.7142, -9.1376)
}

def haversine(lat1, lon1, lat2, lon2):
    if np.isnan(lat1) or np.isnan(lon1):
        return float("inf")
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

closest_place = []
closest_dist = []

for _, row in final_df.iterrows():
    plat, plon = row["lat"], row["lon"]

    best_place = None
    best_dist = float("inf")

    for name, (ilat, ilon) in interest_points.items():
        d = haversine(plat, plon, ilat, ilon)
        if d < best_dist:
            best_dist = d
            best_place = name

    closest_place.append(best_place)
    closest_dist.append(best_dist)

final_df["closest_place"] = closest_place
final_df["distance_km"] = closest_dist

t = checkpoint("Distâncias calculadas corretamente", t)

# ---------------------------------------------------------
# 10. Final sorting & saving
# ---------------------------------------------------------

print("[10/10] A gerar ficheiro final parking_selection.csv...")

final_df = final_df.sort_values(["distance_km", "variance_percent"], ascending=[True, False])
final_df.to_csv("parking_selection.csv", index=False)

t = checkpoint("CSV final gerado", t)

print("\n==============================================")
print("       PROCESSO CONCLUÍDO COM SUCESSO!")
print("       -> Resultado: parking_selection.csv")
print("==============================================\n")
