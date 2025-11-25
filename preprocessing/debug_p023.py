import pandas as pd
import numpy as np

PARKS_FILE = "data/parques_coords.csv"
STREETS_FILE = "traffic_score_by_street.csv"
TRAFFIC_SCORES_FILE = "data/traffic_score_by_hour.csv"

import json

print("Loading data...", flush=True)
df_parks = pd.read_csv(PARKS_FILE)
df_streets = pd.read_csv(STREETS_FILE)
df_traffic = pd.read_csv(TRAFFIC_SCORES_FILE)

def extract_coords(pos_str):
    try:
        data = json.loads(pos_str)
        lon, lat = data["coordinates"]
        return lat, lon
    except:
        return np.nan, np.nan

df_parks["latitude"], df_parks["longitude"] = zip(*df_parks["position"].apply(extract_coords))

# Get P023 coords
p023 = df_parks[df_parks["id_parque"] == "P023"].iloc[0]
print(f"P023 Coords: {p023['latitude']}, {p023['longitude']}", flush=True)

# Get unique streets
unique_streets = df_streets.groupby("local")[["latitude", "longitude"]].mean().reset_index()

# Find closest street
dists = np.sqrt(
    (unique_streets["latitude"] - p023["latitude"])**2 + 
    (unique_streets["longitude"] - p023["longitude"])**2
)
closest_idx = dists.idxmin()
closest_street = unique_streets.iloc[closest_idx]["local"]
print(f"Closest Street to P023: {closest_street}", flush=True)

