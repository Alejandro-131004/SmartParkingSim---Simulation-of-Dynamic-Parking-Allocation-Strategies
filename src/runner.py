import yaml
import json
import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "digital_twin_sim"))

from model import run_once
from policies import make_policy

try:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator
except ImportError:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)

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
    
    df_comb.to_csv(f"reports/{filename}", index=False)
    print(f"[Data] CSV exported to reports/{filename}")

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------

def save_simple_plot(log_data, forecast_df, filename="simple_plot.png", label="Real"):
    """
    Simplified Plot: Predicted vs Real Occupancy only.
    No thresholds, no clutter.
    """
    df = pd.DataFrame(log_data)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df["t"], unit="m")
    
    plt.figure(figsize=(14, 6)) # Shorter height (single plot)
    
    limit = start_date + pd.to_timedelta(48, unit="h") # Zoom to first 48h for clarity? Or full?
    # User said "the real vs predicted". Usually implies full range or zoomed.
    # Let's show full range if it's a monthly simulation, but 48h is better for detail.
    # Actually, previous plot limited to 48h. I'll stick to 48h for clarity unless requested otherwise.
    mask = times < limit
    fc_mask = forecast_df.index < limit
    
    # 1. Predicted (Forecast)
    plt.plot(forecast_df.index[fc_mask], forecast_df.loc[fc_mask, "occupancy_pred"], 
             label="Predicted (Forecast)", color="green", linestyle="--", linewidth=1.5, alpha=0.8)
             
    # 2. Real (Simulation Result)
    plt.plot(times[mask], df[mask]["occ"] * 180, 
             label=f"Real ({label})", color="blue", linewidth=2)
             
    plt.title(f"Real vs Predicted Occupancy ({label})")
    plt.xlabel("Time")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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
    
    # Forecast
    plt.plot(forecast_df.index[fc_mask], forecast_df.loc[fc_mask, "occupancy_pred"], 
             label="Predicted Demand", color="green", linestyle=":", linewidth=2, alpha=0.6)
    
    # Real 1 (e.g. Classic)
    plt.plot(times[mask], df_1[mask]["occ"] * 180, 
             label=label_1, color="gray", linestyle="-", linewidth=1.5, alpha=0.7)
             
    # Real 2 (e.g. AI)
    plt.plot(times[mask], df_2[mask]["occ"] * 180, 
             label=label_2, color="blue", linewidth=2)
    
    plt.title(f"Performance Comparison: {label_1} vs {label_2}")
    plt.ylabel("Vehicles")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] Simplified Comparison PNG saved to reports/{filename}")
    plt.close()


# ---------------------------------------------------------
# TWIN ANALYSIS HELPER
# ---------------------------------------------------------
def run_twin_analysis(cfg, args, test_policy_type, forecaster_type, label_test, scenarios=None):
    if scenarios is None:
        scenarios = {"Base": 1.0, "2x_Demand": 2.0, "3x_Demand": 3.0}

    lot_capacity = cfg["lots"][0]["capacity"]
    
    for sc_name, sc_mult in scenarios.items():
        print(f"\n>>> Running Scenario: {sc_name} (Multiplier: {sc_mult}x)")
        try: sys.stdout.flush() 
        except: pass
        
        cfg["demand_multiplier"] = sc_mult
        
        # 1. Benchmark: STATIC
        print(f"    [{sc_name}] Running Benchmark: STATIC (Env: {forecaster_type})...")
        sim_s = OfflineDigitalTwinSimulator(cfg)
        log_s, df_f = sim_s.run(make_policy("static"), forecaster_type=forecaster_type)
        rev_s, price_s, occ_s, rej_s, served_s = calculate_metrics(log_s, capacity=lot_capacity)
        
        # 2. Test Policy
        print(f"    [{sc_name}] Running Test: {label_test.upper()}...")
        sim_d = OfflineDigitalTwinSimulator(cfg)
        
        # Configure Policy
        policy = make_policy(test_policy_type)
        if hasattr(policy, "target") and args.target is not None: policy.target = args.target
        if hasattr(policy, "k") and args.k is not None: policy.k = args.k
        
        log_d, _ = sim_d.run(policy, forecaster_type=forecaster_type)
        rev_d, price_d, occ_d, rej_d, served_d = calculate_metrics(log_d, capacity=lot_capacity)
        
        # Calculate Delta
        if served_s > 0:
             delta_demand_pct = ((served_d - served_s) / served_s) * 100.0
        else:
             delta_demand_pct = 0.0

        print("\n" + "=" * 94)
        print(f"SCENARIO: {sc_name:<10} | {'METRIC':<18} | {'STATIC':<10} | {label_test.upper():<12} | {'DELTA':<8}")
        print("-" * 94)
        print(f"{' ':24} | {'Revenue (€)':<18} | {rev_s:10.2f} | {rev_d:10.2f} | {rev_d - rev_s:+8.2f}")
        print(f"{' ':24} | {'Avg Price (€)':<18} | {price_s:10.2f} | {price_d:10.2f} | {price_d - price_s:+8.2f}")
        print(f"{' ':24} | {'Avg Occ (%)':<18} | {occ_s*100:10.1f} | {occ_d*100:10.1f} | {(occ_d - occ_s)*100:+8.1f}")
        print(f"{' ':24} | {'Rejection (%)':<18} | {rej_s:10.1f} | {rej_d:10.1f} | {rej_d - rej_s:+8.1f}") 
        print(f"{' ':24} | {'Demand Delta(%)':<18} | {'---':^10} | {delta_demand_pct:+10.1f} | {' ':>8}")
        print("=" * 94 + "\n")
        
        suffix = f"_{label_test}_{sc_name}"
        # GENERATE SIMPLE PLOT (Real vs Predicted)
        # For this analysis, "Real" is the Test Policy Result.
        save_simple_plot(log_d, df_f, filename=f"simple_plot{suffix}.png", label=label_test)
        
        save_csv_results(log_s, log_d, df_f, filename=f"comparison_data{suffix}.csv",
                         label_1="Static", label_2=label_test)

# ---------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=int, choices=[1, 2, 3], default=None,
        help="1=Classic, 2=AI, 3=Battle(Classic vs AI)"
    )
    
    # --- ARGUMENTS FOR SCENARIO RUNNER ---
    parser.add_argument("--target", type=float, default=None, help="Target Occupancy (0.0 - 1.0)")
    parser.add_argument("--k", type=float, default=None, help="Controller Gain K")
    parser.add_argument("--p_min", type=float, default=None, help="Minimum Price")
    parser.add_argument("--silent", action="store_true", help="Output only JSON for automation")
    # -------------------------------------

    args = parser.parse_args()

    # Load Configuration
    config_path = "config/parking_P023.yaml"
    if not os.path.exists(config_path):
        config_path = "../config/parking_P023.yaml"
    
    try:
        cfg = yaml.safe_load(open(config_path))
    except Exception as e:
        if not args.silent:
            print(f"[Error] Failed to load config: {e}")
        return

    # Determine Execution Mode
    mode = args.mode
    
    if mode is None:
        if args.silent:
            mode = 1
        else:
            print("\n================ Execution Mode ================")
            print("1 - Classic Analysis (Static vs Integral Control)")
            print("2 - AI Analysis (Static vs MCTS/Chronos)")
            print("3 - Battle Mode (Classic vs AI)")
            try:
                choice = input("\nSelect (1, 2 or 3): ").strip()
                if choice not in ("1", "2", "3"):
                    mode = 1
                else:
                    mode = int(choice)
            except:
                mode = 1

    # Apply overrides
    if args.target is not None: cfg["policy"]["target_occupancy"] = args.target
    if args.k is not None: cfg["policy"]["k"] = args.k
    if args.p_min is not None: cfg["policy"]["p_min"] = args.p_min

    # ---------------------------------------------------------
    # MODE 1 — Classic Analysis
    # ---------------------------------------------------------
    if mode == 1:
        if not args.silent:
            print("\n[MODE 1] Classic Analysis: Static vs Integral Control (Balanced)\n")
        
        run_twin_analysis(cfg, args, 
                          test_policy_type="dynpricing", 
                          forecaster_type="statistical", 
                          label_test="Classic")

    # ---------------------------------------------------------
    # MODE 2 — AI Analysis
    # ---------------------------------------------------------
    elif mode == 2:
        if not args.silent:
            print("\n[MODE 2] AI Analysis: Static vs MCTS Agent (Chronos)\n")
        
        run_twin_analysis(cfg, args, 
                          test_policy_type="mcts", 
                          forecaster_type="chronos", 
                          label_test="AI")

    # ---------------------------------------------------------
    # MODE 3 — Battle Mode (Classic vs AI)
    # ---------------------------------------------------------
    elif mode == 3:
        if not args.silent:
            print("\n[MODE 3] Battle Mode: Classic (Statistical) vs AI (Chronos + MCTS)\n")

        lot_capacity = cfg["lots"][0]["capacity"]
        scenarios = {"3x_Demand": 3.0}
        
        for sc_name, sc_mult in scenarios.items():
            print(f"\n>>> Running Scenario: {sc_name} (Multiplier: {sc_mult}x)")
            try: sys.stdout.flush() 
            except: pass
            
            cfg["demand_multiplier"] = sc_mult
            
            # --- RUN 1: CLASSIC (Benchmark) ---
            print(f"    [{sc_name}] Running Contender 1: CLASSIC (Using Chronos Env for Fairness)...")
            sim_c = OfflineDigitalTwinSimulator(cfg)
            pol_c = make_policy("balanced")
            if args.target: pol_c.target = args.target
            if args.k: pol_c.k = args.k
            
            # CRITICAL FIX: Use 'chronos' for environment generation so comparisons are fair
            log_c, df_f_c = sim_c.run(pol_c, forecaster_type="chronos")
            rev_c, price_c, occ_c, rej_c, served_c = calculate_metrics(log_c, capacity=lot_capacity)
            
            # --- RUN 2: AI (Challenger) ---
            print(f"    [{sc_name}] Running Contender 2: AI (Chronos + MCTS)...")
            sim_ai = OfflineDigitalTwinSimulator(cfg)
            pol_ai = make_policy("mcts")
            
            log_ai, df_f_ai = sim_ai.run(pol_ai, forecaster_type="chronos")
            rev_ai, price_ai, occ_ai, rej_ai, served_ai = calculate_metrics(log_ai, capacity=lot_capacity)
            
             # Calculate Delta
            if served_c > 0:
                 delta_demand_pct = ((served_ai - served_c) / served_c) * 100.0
            else:
                 delta_demand_pct = 0.0

            # Output
            print("\n" + "=" * 94)
            print(f"SCENARIO: {sc_name:<10} | {'METRIC':<18} | {'CLASSIC':<10} | {'AI':<12} | {'DELTA':<8}")
            print("-" * 94)
            print(f"{' ':24} | {'Revenue (€)':<18} | {rev_c:10.2f} | {rev_ai:10.2f} | {rev_ai - rev_c:+8.2f}")
            print(f"{' ':24} | {'Avg Price (€)':<18} | {price_c:10.2f} | {price_ai:10.2f} | {price_ai - price_c:+8.2f}")
            print(f"{' ':24} | {'Avg Occ (%)':<18} | {occ_c*100:10.1f} | {occ_ai*100:10.1f} | {(occ_ai - occ_c)*100:+8.1f}")
            print(f"{' ':24} | {'Rejection (%)':<18} | {rej_c:10.1f} | {rej_ai:10.1f} | {rej_ai - rej_c:+8.1f}") 
            print(f"{' ':24} | {'Demand Delta(%)':<18} | {'---':^10} | {delta_demand_pct:+10.1f} | {' ':>8}")
            print("=" * 94 + "\n")
            
            suffix = f"_Battle_{sc_name}"
            # Uses simplified comparison plot (Forecast vs Classic vs AI)
            save_comparison_plot_simple(log_c, log_ai, df_f_ai, filename=f"comparison_simple{suffix}.png", 
                             label_1="Classic", label_2="AI")
            
            save_csv_results(log_c, log_ai, df_f_ai, filename=f"comparison_data{suffix}.csv",
                             label_1="Classic", label_2="AI")


if __name__ == "__main__":
    main()
