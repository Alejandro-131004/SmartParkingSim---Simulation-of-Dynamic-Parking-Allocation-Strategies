import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------
def calculate_metrics(occ_log, capacity):
    """
    Calculates simulation metrics.
    metric: Demand Delta (%)
      > 0: Induced Demand (Gain)
      < 0: Lost Customers (Loss)
    """
    df = pd.DataFrame(occ_log)
    if df.empty:
        return 0, 0, 0, 0, 0
    
    avg_price = df["price"].mean()
    avg_occ = df["occ"].mean()
    
    # Each step corresponds to 5 minutes
    time_step_hours = 5.0 / 60.0
    
    revenue = (df["occ"] * capacity * df["price"] * time_step_hours).sum()
    
    total_demand = df["base_entries"].sum()
    total_served = df["actual_entries"].sum()
    
    # METRIC 1: REJECTION % (Saturation)
    # What % of potential demand was lost due to capacity/price?
    if total_demand > 0:
        rejection_pct = ((total_demand - total_served) / total_demand) * 100.0
    else:
        rejection_pct = 0.0
        
    # Total Served is returned for Delta calculation in main loop
    return revenue, avg_price, avg_occ, rejection_pct, total_served

# ---------------------------------------------------------
# DATA EXPORT
# ---------------------------------------------------------
def save_csv_results(static_log, dynamic_log, forecast_df, filename="comparison_data.csv",
                     label_1="Static", label_2="Dynamic"):
    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    
    df_comb = pd.DataFrame()
    df_comb["datetime"] = start_date + pd.to_timedelta(df_s["t"], unit="m")
    df_comb[f"{label_1}_occ"] = df_s["occ"].values * 180
    df_comb[f"{label_2}_occ"] = df_d["occ"].values * 180
    df_comb[f"{label_1}_price"] = df_s["price"].values
    df_comb[f"{label_2}_price"] = df_d["price"].values
    df_comb["traffic_score"] = df_d["traffic"].values
    
    df_comb["base_entries"] = df_d["base_entries"].values
    df_comb["actual_entries"] = df_d["actual_entries"].values
    df_comb["lost_entries_step"] = df_comb["base_entries"] - df_comb["actual_entries"]
    
    os.makedirs("reports", exist_ok=True)
    df_comb.to_csv(f"reports/{filename}", index=False)
    print(f"[Data] CSV exported to reports/{filename}")

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
def save_simple_plot(log_data, forecast_df, filename="simple_plot.png", label="Real"):
    """
    Simplified Plot: Predicted vs Real Occupancy only.
    FIX: Scales forecast correctly for plotting.
    """
    df = pd.DataFrame(log_data)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df["t"], unit="m")
    
    plt.figure(figsize=(14, 6))
    
    limit = start_date + pd.to_timedelta(48, unit="h") 
    mask = times < limit
    fc_mask = forecast_df.index < limit
    
    # --- PLOT FIX: Scale Forecast ---
    # Forecast is usually 0.0-1.0 (Ratio). We need to multiply by capacity (180).
    # We detect scale similar to the simulation fix.
    
    # Get max prediction value to check scale
    max_pred = forecast_df.loc[fc_mask, "occupancy_pred"].max()
    capacity = 180 # Hardcoded or passed as arg. Assuming 180 for P023.
    
    y_pred = forecast_df.loc[fc_mask, "occupancy_pred"]
    
    if max_pred <= 50.0:
        # It's a ratio, scale it up
        y_pred = y_pred * capacity
    
    # 1. Predicted (Forecast)
    plt.plot(forecast_df.index[fc_mask], y_pred, 
             label="Predicted (Forecast)", color="green", linestyle="--", linewidth=1.5, alpha=0.8)
             
    # 2. Real (Simulation Result)
    # occ in log is ratio (0-1), so we multiply by 180
    plt.plot(times[mask], df[mask]["occ"] * capacity, 
             label=f"Real ({label})", color="blue", linewidth=2)
             
    plt.title(f"Real vs Predicted Occupancy ({label})")
    plt.xlabel("Time")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] Simplified PNG saved to reports/{filename}")
    plt.close()

def save_comparison_plot_simple(log_1, log_2, forecast_df, filename="comparison_simple.png", 
                                label_1="Classic", label_2="AI"):
    """
    Simplified Comparison Plot: Forecast vs Real 1 vs Real 2.
    """
    df_1 = pd.DataFrame(log_1)
    df_2 = pd.DataFrame(log_2)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df_1["t"], unit="m")
    
    plt.figure(figsize=(14, 7))
    
    limit = start_date + pd.to_timedelta(48, unit="h")
    mask = times < limit
    fc_mask = forecast_df.index < limit
    
    # --- PLOT FIX ---
    max_pred = forecast_df.loc[fc_mask, "occupancy_pred"].max()
    capacity = 180
    y_pred = forecast_df.loc[fc_mask, "occupancy_pred"]
    if max_pred <= 50.0:
        y_pred = y_pred * capacity

    # Forecast
    plt.plot(forecast_df.index[fc_mask], y_pred, 
             label="Predicted Demand", color="green", linestyle=":", linewidth=2, alpha=0.6)
    
    # Real 1
    plt.plot(times[mask], df_1[mask]["occ"] * capacity, 
             label=label_1, color="gray", linestyle="-", linewidth=1.5, alpha=0.7)
             
    # Real 2
    plt.plot(times[mask], df_2[mask]["occ"] * capacity, 
             label=label_2, color="blue", linewidth=2)
    
    plt.title(f"Performance Comparison: {label_1} vs {label_2}")
    plt.ylabel("Vehicles")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] Simplified Comparison PNG saved to reports/{filename}")
    plt.close()