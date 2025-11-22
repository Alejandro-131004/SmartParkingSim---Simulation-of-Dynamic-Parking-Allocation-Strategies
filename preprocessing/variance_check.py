import pandas as pd

# Load dataset
#file_path = "/Users/franciscamihalache/Desktop/SmartParkingSim---Simulation-of-Dynamic-Parking-Allocation-Strategies/dataset_fluxos_hourly_2022.csv"
file_path = "data/dataset_fluxos_hourly_2022.csv"
df = pd.read_csv(file_path)

# Sort by parking lot and time
df = df.sort_values(["parking_lot", "date", "hour"]).reset_index(drop=True)

# Function to check if a parking lot has >20 consecutive hours with same occupancy
def has_long_constant_period(group, max_hours_same=20):
    # Detect changes in occupancy
    group["change"] = group["occupancy"].diff().ne(0).astype(int)
    group["segment_id"] = group["change"].cumsum()
    counts = group["segment_id"].value_counts()
    longest_run = counts.max()
    return longest_run > max_hours_same

# Identify which parking lots violate the rule
bad_lots = []
for lot, group in df.groupby("parking_lot"):
    if has_long_constant_period(group):
        bad_lots.append(lot)

# Filter out those parking lots entirely
df_filtered = df[~df["parking_lot"].isin(bad_lots)]

print(f"Removed {len(bad_lots)} parking lots with >20h constant occupancy:")
print(bad_lots)

# Compute variance and relative variance only for valid lots
variance_df = (
    df_filtered.groupby("parking_lot")["occupancy"]
    .var()
    .reset_index(name="variance")
)
mean_df = (
    df_filtered.groupby("parking_lot")["occupancy"]
    .mean()
    .reset_index(name="mean_occupancy")
)
variance_df = variance_df.merge(mean_df, on="parking_lot")
variance_df["variance_percent"] = (variance_df["variance"] / variance_df["mean_occupancy"]) * 100

# Sort descending
variance_df = variance_df.sort_values("variance_percent", ascending=False)

print(variance_df.head(10))
print(variance_df[variance_df["parking_lot"] == "P064"])
