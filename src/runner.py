import yaml
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PATH CONFIGURATION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add digital_twin_sim folder to path so internal imports work
sys.path.append(os.path.join(current_dir, 'digital_twin_sim'))

# ==========================================
# IMPORTS
# ==========================================
from model import run_once
from policies import make_policy

# Try importing the simulator with a fallback mechanism
try:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator
except ImportError:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)

# ==========================================
# HELPER FUNCTIONS (Mode 3)
# ==========================================

def calculate_metrics(occ_log, capacity=180):
    """Calculates key metrics: Revenue, Avg Price, Avg Occupancy."""
    df = pd.DataFrame(occ_log)
    if df.empty:
        return 0, 0, 0
    
    avg_price = df['price'].mean()
    avg_occ = df['occ'].mean()
    
    # Revenue Estimation: Sum(Occupied Spots * Price * TimeDuration)
    # Sampling interval is 5 minutes = 5/60 hours
    time_step_hours = 5.0 / 60.0
    occupied_spots = df['occ'] * capacity
    revenue_step = occupied_spots * df['price'] * time_step_hours
    total_revenue = revenue_step.sum()
    
    return total_revenue, avg_price, avg_occ

def save_comparison_plot(static_log, dynamic_log, forecast_df, filename="comparison_plot.png"):
    """Generates the Static vs Dynamic comparison plot."""
    df_static = pd.DataFrame(static_log)
    df_dynamic = pd.DataFrame(dynamic_log)
    
    # Create time axis based on forecast start
    start_date = forecast_df.index[0]
    df_static['datetime'] = start_date + pd.to_timedelta(df_static['t'], unit='m')
    df_dynamic['datetime'] = start_date + pd.to_timedelta(df_dynamic['t'], unit='m')

    plt.figure(figsize=(14, 8))
    
    # --- Subplot 1: Occupancy ---
    plt.subplot(2, 1, 1)
    # Zoom in on first 48h to see the curve clearly
    limit_time = start_date + pd.to_timedelta(48, unit='h') 
    mask_s = df_static['datetime'] < limit_time
    mask_d = df_dynamic['datetime'] < limit_time
    
    plt.plot(df_static[mask_s]['datetime'], df_static[mask_s]['occ'] * 180, 
             label='Static Pricing (Benchmark)', color='gray', linestyle='--', alpha=0.6)
    
    plt.plot(df_dynamic[mask_d]['datetime'], df_dynamic[mask_d]['occ'] * 180, 
             label='Dynamic Pricing (Your Model)', color='blue', linewidth=2)
            
    plt.axhline(y=180*0.8, color='green', linestyle=':', label='Target (80%)')
    plt.title("Occupancy: Static vs Dynamic (First 48h)")
    plt.ylabel("Cars Parked")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Subplot 2: Price ---
    plt.subplot(2, 1, 2)
    plt.plot(df_static[mask_s]['datetime'], df_static[mask_s]['price'], 
             label='Static Price', color='gray', linestyle='--')
    plt.step(df_dynamic[mask_d]['datetime'], df_dynamic[mask_d]['price'], 
             where='post', label='Dynamic Price', color='red')
             
    plt.title("Dynamic Price Strategy")
    plt.ylabel("Price (€/h)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] Plot saved to reports/{filename}")
    plt.close()

def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    config_path = "config/parking_P023.yaml"
    if not os.path.exists(config_path):
        # Fallback if running from src root
        config_path = "../config/parking_P023.yaml"
        
    try:
        cfg = yaml.safe_load(open(config_path))
    except FileNotFoundError:
        print("Error: Configuration file not found. Run from project root.")
        return

    print("\n================ Execution Mode ================")
    print("1 - Normal Simulation (SimPy - Original)")
    print("3 - Offline Digital Twin (Comparative: Static vs Dynamic)")
    
    choice = input("\nSelect execution mode (1 or 3): ").strip()
    if choice == "": choice = "3"
    
    # --- MODE 1: NORMAL SIMULATION ---
    if choice == "1":
        print("\n[MODE 1] Normal Simulation Selected\n")
        policy = make_policy(cfg["policy"])
        if hasattr(policy, "set_elasticities"): policy.set_elasticities(cfg.get("elasticity", {}))
        if hasattr(policy, "set_event_calendar"): policy.set_event_calendar(cfg.get("event_calendar", []))

        results = []
        for r in range(cfg["runs"]):
            seed = cfg["seed"] + r
            out = run_once(cfg, policy, seed)
            out["seed"] = seed
            results.append(out)

        pd.DataFrame(results).to_csv("reports/results_normal_sim.csv", index=False)
        print("Saved normal simulation results.\n")
        return

    # --- MODE 3: COMPARATIVE DIGITAL TWIN ---
    elif choice == "3":
        print("\n[MODE 3] Offline Digital Twin - Comparative Analysis\n")
        
        sim = OfflineDigitalTwinSimulator(cfg)
        
        # 1. Benchmark (Static)
        print(">>> Running Benchmark: STATIC Policy...")
        StaticPolicyClass = make_policy("static")
        log_static, df_forecast = sim.run(StaticPolicyClass)
        rev_s, price_s, occ_s = calculate_metrics(log_static)
        
        # 2. Proposal (Dynamic)
        print("\n>>> Running Proposal: DYNAMIC Policy...")
        policy_name = cfg.get("policy", "dynpricing")
        DynamicPolicyClass = make_policy(policy_name)
        log_dynamic, _ = sim.run(DynamicPolicyClass) 
        rev_d, price_d, occ_d = calculate_metrics(log_dynamic)

        # 3. Comparison Table
        print("\n" + "="*60)
        print(f"{'METRIC':<20} | {'STATIC':<12} | {'DYNAMIC':<12} | {'DELTA':<10}")
        print("-" * 60)
        print(f"{'Revenue (€)':<20} | {rev_s:12.2f} | {rev_d:12.2f} | {rev_d - rev_s:+10.2f}")
        print(f"{'Avg Price (€)':<20} | {price_s:12.2f} | {price_d:12.2f} | {price_d - price_s:+10.2f}")
        print(f"{'Avg Occupancy (%)':<20} | {occ_s*100:12.1f} | {occ_d*100:12.1f} | {(occ_d - occ_s)*100:+10.1f}")
        print("="*60 + "\n")

        # 4. Save Data and Plot
        analysis = {
            "static": {"revenue": rev_s, "avg_price": price_s, "avg_occ": occ_s, "log": log_static},
            "dynamic": {"revenue": rev_d, "avg_price": price_d, "avg_occ": occ_d, "log": log_dynamic}
        }
        with open("reports/offline_digital_twin.json", "w") as f:
            json.dump(convert_timestamps(analysis), f, indent=2)

        save_comparison_plot(log_static, log_dynamic, df_forecast)
        return

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()