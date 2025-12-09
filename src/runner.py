import yaml
import json
import os
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'digital_twin_sim'))

from model import run_once
from policies import make_policy

try:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator
except ImportError:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)

def calculate_metrics(occ_log, capacity=180):
    df = pd.DataFrame(occ_log)
    if df.empty: return 0, 0, 0, 0
    
    avg_price = df['price'].mean()
    avg_occ = df['occ'].mean()
    
    time_step_hours = 5.0 / 60.0
    revenue = (df['occ'] * capacity * df['price'] * time_step_hours).sum()
    
    total_demand = df['base_entries'].sum()
    total_served = df['actual_entries'].sum()
    
    if total_demand > 0:
        rejection_rate = (1 - (total_served / total_demand)) * 100.0
    else:
        rejection_rate = 0.0
        
    return revenue, avg_price, avg_occ, rejection_rate

def save_csv_results(static_log, dynamic_log, forecast_df, filename="comparison_data.csv"):
    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    
    df_comb = pd.DataFrame()
    df_comb['datetime'] = start_date + pd.to_timedelta(df_s['t'], unit='m')
    df_comb['static_occ'] = df_s['occ'].values * 180
    df_comb['dynamic_occ'] = df_d['occ'].values * 180
    df_comb['static_price'] = df_s['price'].values
    df_comb['dynamic_price'] = df_d['price'].values
    df_comb['traffic_score'] = df_d['traffic'].values 
    
    df_comb['base_entries'] = df_d['base_entries'].values
    df_comb['actual_entries'] = df_d['actual_entries'].values
    # Diferença: Clientes Perdidos nesse intervalo
    df_comb['lost_entries_step'] = df_comb['base_entries'] - df_comb['actual_entries']
    
    df_comb.to_csv(f"reports/{filename}", index=False)
    print(f"[Data] CSV exportado para reports/{filename}")

def save_interactive_plot(static_log, dynamic_log, forecast_df, filename="comparison_interactive.html"):
    if not PLOTLY_AVAILABLE: return

    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df_s['t'], unit='m')

    # 4 Subplots agora
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03,
                        subplot_titles=("Ocupação (Carros)", "Preço (€/h)", "Tráfego Urbano", "Clientes Perdidos (Fluxo/5min)"))

    # 1. Occupancy
    fig.add_trace(go.Scatter(x=times, y=df_s['occ']*180, name='Static', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=df_d['occ']*180, name='Dynamic', line=dict(color='blue')), row=1, col=1)
    fig.add_hline(y=180*0.8, line_dash="dot", line_color="green", row=1, col=1)

    # 2. Price
    fig.add_trace(go.Scatter(x=times, y=df_d['price'], name='Price', line=dict(color='red', shape='hv')), row=2, col=1)

    # 3. Traffic
    fig.add_trace(go.Scatter(x=times, y=df_d['traffic'], name='Traffic', line=dict(color='orange'), fill='tozeroy'), row=3, col=1)

    # 4. Lost Customers (Flow)
    lost = df_d['base_entries'] - df_d['actual_entries']
    fig.add_trace(go.Bar(x=times, y=lost, name='Lost Customers', marker_color='purple'), row=4, col=1)

    fig.update_layout(height=1000, title_text="Digital Twin: Full Analysis (Inputs/Exits/Traffic)", hovermode="x unified")
    fig.write_html(f"reports/{filename}")
    print(f"[Visual] Gráfico Interativo guardado em reports/{filename}")

def save_static_plot(static_log, dynamic_log, forecast_df, filename="comparison_plot.png"):
    df_s = pd.DataFrame(static_log)
    df_d = pd.DataFrame(dynamic_log)
    start_date = forecast_df.index[0]
    times = start_date + pd.to_timedelta(df_s['t'], unit='m')

    plt.figure(figsize=(14, 12))
    
    limit = start_date + pd.to_timedelta(48, unit='h')
    mask = times < limit
    
    # 1. Occupancy
    plt.subplot(4, 1, 1)
    plt.plot(times[mask], df_s[mask]['occ']*180, label='Static', color='gray', linestyle='--')
    plt.plot(times[mask], df_d[mask]['occ']*180, label='Dynamic', color='blue')
    plt.axhline(y=180*0.8, color='green', linestyle=':', label='Target')
    plt.ylabel("Carros")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Price
    plt.subplot(4, 1, 2)
    plt.step(times[mask], df_d[mask]['price'], where='post', label='Price', color='red')
    plt.ylabel("Preço (€)")
    plt.grid(True, alpha=0.3)
    
    # 3. Traffic
    plt.subplot(4, 1, 3)
    plt.plot(times[mask], df_d[mask]['traffic'], label='Traffic', color='orange')
    plt.ylabel("Tráfego")
    plt.grid(True, alpha=0.3)

    # 4. Lost Customers
    plt.subplot(4, 1, 4)
    lost = df_d['base_entries'] - df_d['actual_entries']
    plt.bar(times[mask], lost[mask], label='Lost Customers', color='purple', width=0.003)
    plt.ylabel("Recusas (5min)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"reports/{filename}")
    print(f"[Visual] PNG guardado em reports/{filename}")
    plt.close()

def convert_timestamps(obj):
    if isinstance(obj, dict): return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_timestamps(v) for v in obj]
    elif hasattr(obj, "isoformat"): return obj.isoformat()
    return obj

def main():
    config_path = "config/parking_P023.yaml"
    if not os.path.exists(config_path): config_path = "../config/parking_P023.yaml"
    
    try: cfg = yaml.safe_load(open(config_path))
    except: return

    print("\n================ Execution Mode ================")
    print("1 - Normal Simulation")
    print("3 - Offline Digital Twin (Traffic + Exit Friction)")
    
    choice = input("\nSelect (1 or 3): ").strip() or "3"
    
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

    elif choice == "3":
        print("\n[MODE 3] Offline Digital Twin - Traffic Aware\n")
        sim = OfflineDigitalTwinSimulator(cfg)
        
        print(">>> Running Benchmark: STATIC...")
        log_s, df_f = sim.run(make_policy("static"))
        rev_s, price_s, occ_s, rej_s = calculate_metrics(log_s)
        
        print("\n>>> Running Proposal: DYNAMIC...")
        log_d, _ = sim.run(make_policy(cfg.get("policy", "dynpricing")))
        rev_d, price_d, occ_d, rej_d = calculate_metrics(log_d)

        print("\n" + "="*65)
        print(f"{'METRIC':<20} | {'STATIC':<12} | {'DYNAMIC':<12} | {'DELTA':<10}")
        print("-" * 65)
        print(f"{'Revenue (€)':<20} | {rev_s:12.2f} | {rev_d:12.2f} | {rev_d - rev_s:+10.2f}")
        print(f"{'Avg Price (€)':<20} | {price_s:12.2f} | {price_d:12.2f} | {price_d - price_s:+10.2f}")
        print(f"{'Avg Occupancy (%)':<20} | {occ_s*100:12.1f} | {occ_d*100:12.1f} | {(occ_d - occ_s)*100:+10.1f}")
        print(f"{'Lost Customers (%)':<20} | {rej_s:12.1f} | {rej_d:12.1f} | {rej_d - rej_s:+10.1f}")
        print("="*65 + "\n")

        analysis = {
            "static": {"revenue": rev_s, "avg_price": price_s, "avg_occ": occ_s, "rejection": rej_s},
            "dynamic": {"revenue": rev_d, "avg_price": price_d, "avg_occ": occ_d, "rejection": rej_d}
        }
        with open("reports/offline_digital_twin.json", "w") as f:
            json.dump(convert_timestamps(analysis), f, indent=2)

        save_static_plot(log_s, log_d, df_f)
        save_interactive_plot(log_s, log_d, df_f)
        save_csv_results(log_s, log_d, df_f)

if __name__ == "__main__":
    main()