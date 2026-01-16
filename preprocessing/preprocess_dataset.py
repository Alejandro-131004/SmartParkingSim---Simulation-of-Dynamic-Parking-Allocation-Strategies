import pandas as pd
import numpy as np
import ast
from concurrent.futures import ProcessPoolExecutor
import sys

def extract_coordinates(coord_str):
    try:
        if isinstance(coord_str, str):
            coord_dict = ast.literal_eval(coord_str)
        else:
            coord_dict = coord_str
        if isinstance(coord_dict, dict) and "coordinates" in coord_dict:
            lon, lat = coord_dict["coordinates"]
            return lat, lon
    except Exception:
        pass
    return None, None


def get_weighted_traffic_score(ts, lat, lon, traffic_df, top_n=5, max_dist_km=2.0):
    # Ensure date/hour matching
    target_date = pd.to_datetime(ts).date()
    target_hour = pd.to_datetime(ts).hour

    traffic_hour = traffic_df[
        (traffic_df["date"] == target_date) &
        (traffic_df["hour"] == target_hour)
    ]

    if traffic_hour.empty:
        return np.nan

    rads = np.radians

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius (km)
        dlat = rads(lat2 - lat1)
        dlon = rads(lon2 - lon1)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(rads(lat1)) * np.cos(rads(lat2)) * np.sin(dlon / 2) ** 2
        )
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    traffic_hour = traffic_hour.copy()
    traffic_hour["distance_km"] = traffic_hour.apply(
        lambda row: haversine(lat, lon, row["latitude"], row["longitude"]),
        axis=1
    )

    traffic_near = (
        traffic_hour[traffic_hour["distance_km"] <= max_dist_km]
        .sort_values("distance_km")
        .head(top_n)
    )

    if not traffic_near.empty:
        weights = 1 / (traffic_near["distance_km"].replace(0, 0.1) ** 2)
        score = np.average(traffic_near["score"], weights=weights)
        return round(float(score), 2)

    return np.nan


def traffic_score_row(args):
    row, traffic_df = args
    return get_weighted_traffic_score(
        row["timestamp_hour"],
        row["lat"],
        row["lon"],
        traffic_df
    )


def preprocess_parking_data_simplified(
    input_file,
    output_file,
    traffic_score_file,
    nthreads=8
):
    print("\n" + "=" * 70)
    print(" PRE-PROCESSING: OCCUPANCY AND TRAFFIC")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load data and normalize headers
    # -----------------------------------------------------
    print(f"\nLoading: {input_file}")
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fuzzy column matching
    park_col = next((c for c in df.columns if any(x in c for x in ["parque", "name"])), None)
    capacity_col = next((c for c in df.columns if any(x in c for x in ["capacity", "cap"])), None)
    occupancy_col = next((c for c in df.columns if any(x in c for x in ["occupancy", "occ"])), None)
    timestamp_col = next((c for c in df.columns if any(x in c for x in ["timestamp", "date", "time", "dtm"])), None)
    coord_col = next((c for c in df.columns if c in ["coordinates", "position"]), None)

    # Rename for consistency
    df = df.rename(columns={
        park_col: "parking_lot",
        occupancy_col: "occupancy",
        timestamp_col: "timestamp"
    })

    if capacity_col:
        df = df.rename(columns={capacity_col: "capacity"})

    # -----------------------------------------------------
    # 2. Extract coordinates
    # -----------------------------------------------------
    if coord_col:
        coords = df[coord_col].apply(extract_coordinates)
        df["lat"] = coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
        df["lon"] = coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
        print(f"Coordinates extracted from '{coord_col}'")
    else:
        print("Error: Coordinate column not found")
        return

    # -----------------------------------------------------
    # 3. Time normalization
    # -----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_hour"] = df["timestamp"].dt.floor("h")

    if "capacity" not in df.columns:
        df["capacity"] = df.groupby("parking_lot")["occupancy"].transform("max")

    # Drop rows with missing essentials
    df = df.dropna(subset=["parking_lot", "lat", "lon", "timestamp_hour", "occupancy"])

    # -----------------------------------------------------
    # 4. Traffic score computation (parallel)
    # -----------------------------------------------------
    print("🚦 Computing proximity-weighted traffic scores...")
    traffic_df = pd.read_csv(traffic_score_file)
    traffic_df["date"] = pd.to_datetime(traffic_df["date"]).dt.date
    traffic_df["hour"] = pd.to_numeric(traffic_df["hour"], errors="coerce")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        args_list = [(row, traffic_df) for row in df.to_dict("records")]
        df["traffic_score"] = list(
            executor.map(traffic_score_row, args_list, chunksize=100)
        )

    # -----------------------------------------------------
    # 5. Hourly aggregation
    # -----------------------------------------------------
    df["date"] = df["timestamp_hour"].dt.date
    df["hour"] = df["timestamp_hour"].dt.hour

    hourly = (
        df.groupby(
            ["parking_lot", "capacity", "date", "hour"],
            as_index=False
        )
        .agg({
            "occupancy": "mean",
            "traffic_score": "mean",
            "lat": "first",
            "lon": "first"
        })
    )

    hourly["occupancy"] = hourly["occupancy"].round().astype(int)
    hourly["traffic_score"] = hourly["traffic_score"].round(2)

    # -----------------------------------------------------
    # 6. Save output
    # -----------------------------------------------------
    hourly.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDone. Output saved to: {output_file}")
    print(f"Processed records: {len(hourly)}")


if __name__ == "__main__":
    preprocess_parking_data_simplified(
        input_file="data/parques_data/parques-estacionamento-1s-2022.xlsx",
        output_file="parking_occupancy_traffic_2022.csv",
        traffic_score_file="data/traffic/traffic_score_by_hour.csv",
        nthreads=16
    )
