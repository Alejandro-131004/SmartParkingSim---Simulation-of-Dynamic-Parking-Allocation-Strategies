import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
try:
    # Environment drivers
    df_raw = pd.read_csv('data/P023_sim/dataset_P023_simready_updated.csv')
    # Simulation outcomes
    df_sim = pd.read_csv('reports/comparison_data_GrandBattle_3x_Demand.csv')
except FileNotFoundError:
    print("Error: Check folder paths (data/ and reports/) and filenames.")
    exit()

# 2. Data Alignment
df_sim['datetime'] = pd.to_datetime(df_sim['datetime'])
df_raw['datetime'] = pd.to_datetime(df_raw['date']) + pd.to_timedelta(df_raw['hour'], unit='h')
df_combined = pd.merge(df_sim, df_raw, on='datetime')
df_combined['avg_temp'] = (df_combined['temp_min_c'] + df_combined['temp_max_c']) / 2.0

# 3. Correlation Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: Traffic vs Occupancy
axes[0].scatter(df_combined['traffic_score'], df_combined['MCTS_Chronos_occ'], alpha=0.4, color='#1f77b4')
axes[0].set_title("Traffic Congestion vs. Occupancy\n(The Friction Effect)")
axes[0].set_xlabel("Traffic Score")
axes[0].set_ylabel("Occupancy (Vehicles)")
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot B: Temperature vs Occupancy
axes[1].scatter(df_combined['avg_temp'], df_combined['MCTS_Chronos_occ'], alpha=0.4, color='#d62728')
axes[1].set_title("Temperature vs. Occupancy\n(The Comfort/Urgency Effect)")
axes[1].set_xlabel("Average Temperature (°C)")
axes[1].set_ylabel("Occupancy (Vehicles)")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('reports/driver_influence_evidence.png')
print("Proof of influence plot saved to: reports/driver_influence_evidence.png")