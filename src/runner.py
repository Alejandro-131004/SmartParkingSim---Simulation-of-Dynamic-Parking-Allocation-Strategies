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
        return 0, 0, 0, 0
    
    avg_price = df["price"].mean()
    avg_occ = df["occ"].mean()
    
    # Each step corresponds to 5 minutes
    time_step_hours = 5.0 / 60.0
    
    revenue = (df["occ"] * capacity * df["price"] * time_step_hours).sum()
    
    total_demand = df["base_entries"].sum()
    total_served = df["actual_entries"].sum()
    
    if total_demand > 0:
        # NEW LOGIC: (Served - Demand) / Demand
        # Result: +5.0 means we served 5% MORE than baseline (Gain)
        # Result: -5.0 means we served 5% LESS than baseline (Loss)
        demand_delta_pct = ((total_served - total_demand) / total_demand) * 100.0
    else:
        demand_delta_pct = 0.0
        
    return revenue, avg_price, avg_occ, demand_delta_pct
# ---------------------------------------------------------
# DATA EXPORT
# ---------------------------------------------------------
def save_csv_results(static_log, dynamic_log, forecast_df, filename="comparison_data.csv"):
    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    
    df_comb = pd.DataFrame()
    df_comb["datetime"] = start_date + pd.to_timedelta(df_s["t"], unit="m")
    df_comb["static_occ"] = df_s["occ"].values * 180
    df_comb["dynamic_occ"] = df_d["occ"].values * 180
    df_comb["static_price"] = df_s["price"].values
    df_comb["dynamic_price"] = df_d["price"].values
    df_comb["traffic_score"] = df_d["traffic"].values
    
    df_comb["base_entries"] = df_d["base_entries"].values
    df_comb["actual_entries"] = df_d["actual_entries"].values
    df_comb["lost_entries_step"] = df_comb["base_entries"] - df_comb["actual_entries"]
    
    df_comb.to_csv(f"reports/{filename}", index=False)
    print(f"[Data] CSV exported to reports/{filename}")

# ---------------------------------------------------------
# INTERACTIVE PLOT (PLOTLY)
# ---------------------------------------------------------
def save_interactive_plot(static_log, dynamic_log, forecast_df, filename="comparison_interactive.html"):
    if not PLOTLY_AVAILABLE:
        return

    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df_s["t"], unit="m")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(
            "Occupancy (Vehicles)",
            "Price (€/h)",
            "Urban Traffic Score",
            "Lost Customers per Interval (5 min)"
        )
    )

    # 1. Occupancy
    fig.add_trace(
        go.Scatter(x=times, y=df_s["occ"] * 180, name="Static", 
                   line=dict(color="gray", dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=df_d["occ"] * 180, name="Dynamic", 
                   line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_hline(y=180 * 0.8, line_dash="dot", line_color="green", row=1, col=1)

    # 2. Price
    fig.add_trace(
        go.Scatter(x=times, y=df_d["price"], name="Price", 
                   line=dict(color="red", shape="hv")),
        row=2, col=1
    )

    # 3. Traffic
    fig.add_trace(
        go.Scatter(x=times, y=df_d["traffic"], name="Traffic", 
                   line=dict(color="orange"), fill="tozeroy"),
        row=3, col=1
    )

    # 4. Lost Customers
    lost = df_d["base_entries"] - df_d["actual_entries"]
    fig.add_trace(
        go.Bar(x=times, y=lost, name="Lost Customers", marker_color="purple"),
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        title_text="Digital Twin: Full Analysis (Occupancy / Price / Traffic / Demand Loss)",
        hovermode="x unified"
    )

    fig.write_html(f"reports/{filename}")
    print(f"[Visual] Interactive chart saved to reports/{filename}")

# ---------------------------------------------------------
# STATIC PLOT (MATPLOTLIB)
# ---------------------------------------------------------
def save_static_plot(static_log, dynamic_log, forecast_df, filename="comparison_plot.png"):
    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df_s["t"], unit="m")

    plt.figure(figsize=(14, 12))
    
    limit = start_date + pd.to_timedelta(48, unit="h")
    mask = times < limit

    # 1. Occupancy
    plt.subplot(4, 1, 1)
    plt.plot(times[mask], df_s[mask]["occ"] * 180, label="Static", color="gray", linestyle="--")
    plt.plot(times[mask], df_d[mask]["occ"] * 180, label="Dynamic", color="blue")
    plt.axhline(y=180 * 0.8, color="green", linestyle=":", label="Target")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Price
    plt.subplot(4, 1, 2)
    plt.step(times[mask], df_d[mask]["price"], where="post", label="Price", color="red")
    plt.ylabel("Price (€)")
    plt.grid(True, alpha=0.3)

    # 3. Traffic
    plt.subplot(4, 1, 3)
    plt.plot(times[mask], df_d[mask]["traffic"], label="Traffic", color="orange")
    plt.ylabel("Traffic")
    plt.grid(True, alpha=0.3)

    # 4. Lost Customers
    plt.subplot(4, 1, 4)
    lost = df_d["base_entries"] - df_d["actual_entries"]
    plt.bar(times[mask], lost[mask], label="Lost Customers", color="purple", width=0.003)
    plt.ylabel("Declined (5 min)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] PNG saved to reports/{filename}")
    plt.close()

# ---------------------------------------------------------
# TIMESTAMP SANITIZER
# ---------------------------------------------------------
def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj

# ---------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=int, choices=[1, 2], default=None,
        help="1 = Normal Simulation, 2 = Offline Digital Twin"
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
            # Default to Twin mode if running silently without explicit mode
            mode = 2
        else:
            print("\n================ Execution Mode ================")
            print("1 - Normal Simulation")
            print("2 - Offline Digital Twin (Traffic + Exit Friction)")
            try:
                choice = input("\nSelect (1 or 2): ").strip()
                if choice not in ("1", "2"):
                    mode = 1
                else:
                    mode = int(choice)
            except:
                mode = 1

    # ---------------------------------------------------------
    # MODE 1 — Standard simulation
    # ---------------------------------------------------------
    if mode == 1:
        if not args.silent:
            print("\n[MODE 1] Normal Simulation Selected\n")
        
        policy_cfg = cfg.get("policy", {})
        policy = make_policy(policy_cfg)

        if hasattr(policy, "set_elasticities"):
            policy.set_elasticities(cfg.get("elasticity", {}))
        if hasattr(policy, "set_event_calendar"):
            policy.set_event_calendar(cfg.get("event_calendar", []))

        results = []
        for r in range(cfg.get("runs", 1)):
            seed = cfg.get("seed", 42) + r
            out = run_once(cfg, policy, seed)
            out["seed"] = seed
            results.append(out)

        pd.DataFrame(results).to_csv("reports/results_normal_sim.csv", index=False)
        if not args.silent:
            print("Saved normal simulation results.\n")
        return

    # ---------------------------------------------------------
    # MODE 2 — Offline Digital Twin
    # ---------------------------------------------------------
    elif mode == 2:
        if not args.silent:
            print("\n[MODE 2] Offline Digital Twin - Traffic Aware\n")
        
        # --- UPDATE CONFIG WITH CLI ARGUMENTS ---
        if "policy" not in cfg or isinstance(cfg["policy"], str):
             old_val = cfg.get("policy", "dynpricing")
             cfg["policy"] = {"type": old_val} if isinstance(old_val, str) else {}

        # Apply overrides from Scenario Runner
        if args.target is not None:
            cfg["policy"]["target_occupancy"] = args.target
        if args.k is not None:
            cfg["policy"]["k"] = args.k
        if args.p_min is not None:
            cfg["policy"]["p_min"] = args.p_min
        # ----------------------------------------

        sim = OfflineDigitalTwinSimulator(cfg)

        # EXTRACT CAPACITY FROM CONFIG
        lot_capacity = cfg["lots"][0]["capacity"]

        # 1. Benchmark: STATIC
        if not args.silent: print(">>> Running Benchmark: STATIC...")
        log_s, df_f = sim.run(make_policy("static"))
        rev_s, price_s, occ_s, rej_s = calculate_metrics(log_s, capacity=lot_capacity)
        
        # 2. Benchmark: DYNAMIC (Proposal)
        if not args.silent: print("\n>>> Running Proposal: DYNAMIC...")
        
        # --- CORREÇÃO DE NOME DA POLÍTICA ---
        # O nome correto no policies.py é "dynpricing"
        p_type = cfg["policy"].get("type", "dynpricing")
        
        # Se por acaso a config tiver "dynamic" ou "linear", forçamos "dynpricing"
        if p_type in ["dynamic", "linear"]:
            p_type = "dynpricing"

        # Instanciar a política correta
        dyn_policy = make_policy(p_type)

        # Atualizar os parâmetros no objeto da política
        # A classe DynamicPricingBalanced usa 'self.target', 'self.k', etc.
        if "target_occupancy" in cfg["policy"]:
            dyn_policy.target = cfg["policy"]["target_occupancy"]
        if "k" in cfg["policy"]:
            dyn_policy.k = cfg["policy"]["k"]
        if "p_min" in cfg["policy"]:
            dyn_policy.p_min = cfg["policy"]["p_min"]
        
        # Run simulation
        log_d, _ = sim.run(dyn_policy)
        rev_d, price_d, occ_d, rej_d = calculate_metrics(log_d, capacity=lot_capacity)

        # 3. Output Handling
        if args.silent:
            output_data = {
                "target": args.target if args.target is not None else cfg["policy"].get("target_occupancy", 0),
                "k": args.k if args.k is not None else cfg["policy"].get("k", 0),
                "p_min": args.p_min if args.p_min is not None else cfg["policy"].get("p_min", 0),
                "static_revenue": rev_s,
                "dynamic_revenue": rev_d,
                "dynamic_price": price_d,
                "dynamic_occ": occ_d,
                "lost_pct": rej_d
            }
            print(json.dumps(output_data))
        else:
            print("\n" + "=" * 65)
            print(f"{'METRIC':<20} | {'STATIC':<12} | {'DYNAMIC':<12} | {'DELTA':<10}")
            print("-" * 65)
            print(f"{'Revenue (€)':<20} | {rev_s:12.2f} | {rev_d:12.2f} | {rev_d - rev_s:+10.2f}")
            print(f"{'Avg Price (€)':<20} | {price_s:12.2f} | {price_d:12.2f} | {price_d - price_s:+10.2f}")
            print(f"{'Avg Occupancy (%)':<20} | {occ_s*100:12.1f} | {occ_d*100:12.1f} | {(occ_d - occ_s)*100:+10.1f}")
            print(f"{'Demand Change (%)':<20} | {rej_s:+12.1f} | {rej_d:+12.1f} | {rej_d - rej_s:+10.1f}") # Positive = Gain, Negative = Loss
            print("=" * 65 + "\n")

            save_static_plot(log_s, log_d, df_f)
            save_interactive_plot(log_s, log_d, df_f)
            save_csv_results(log_s, log_d, df_f)


if __name__ == "__main__":
    main()
