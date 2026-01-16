import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load raw data (drivers of demand) and simulation results (outcomes)
df_raw = pd.read_csv('data/P023_sim/dataset_P023_simready_updated.csv')
df_sim = pd.read_csv('reports/comparison_data_GrandBattle_3x_Demand.csv')

# 2. Align datasets by time
df_sim['datetime'] = pd.to_datetime(df_sim['datetime'])
# Combine date and hour into a single datetime for merging
df_raw['datetime'] = pd.to_datetime(df_raw['date']) + pd.to_timedelta(df_raw['hour'], unit='h')
df_combined = pd.merge(df_sim, df_raw, on='datetime')

# Create an average temperature column from the raw dataset
df_combined['avg_temp'] = (df_combined['temp_min_c'] + df_combined['temp_max_c']) / 2.0

# 3. Generate the Dual-Panel Evidence Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Traffic Influence
# Shows the 'friction' effect: higher traffic congestion leads to lower realized entries
ax1.scatter(df_combined['traffic_score'], df_combined['Classic_Stat_occ'], alpha=0.5, color='blue')
ax1.set_title("Evidence A: Traffic Congestion Influence")
ax1.set_xlabel("Traffic Congestion Level (Traffic Score)")
ax1.set_ylabel("Simulated Occupancy (Vehicles)")
ax1.grid(True, alpha=0.3)

# Subplot 2: Temperature Influence
# Shows the 'comfort/urgency' effect: occupancy shifts based on thermal discomfort
ax2.scatter(df_combined['avg_temp'], df_combined['Classic_Stat_occ'], alpha=0.5, color='red')
ax2.set_title("Evidence B: Temperature Influence")
ax2.set_xlabel("Average Temperature (°C)")
ax2.set_ylabel("Simulated Occupancy (Vehicles)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/environmental_influence_proof.png")
print("Evidence plot saved to: reports/environmental_influence_proof.png")