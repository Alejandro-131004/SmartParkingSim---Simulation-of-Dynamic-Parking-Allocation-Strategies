
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# CONFIG
DATASET_PATH = "data/P023_sim/dataset_P023_simready_updated.csv"
REPORT_DIR = "reports"
DEFAULT_FILE = "comparison_data_Battle_3x_Demand.csv"

def load_data(csv_name):
    path = os.path.join(REPORT_DIR, csv_name)
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        print(f"Available files in {REPORT_DIR}:")
        for f in os.listdir(REPORT_DIR):
            if f.endswith(".csv"): print(f" - {f}")
        return None
    return pd.read_csv(path)

def load_ground_truth():
    if not os.path.exists(DATASET_PATH):
        print(f"[Warn] Ground Truth dataset not found at {DATASET_PATH}")
        return None
    df = pd.read_csv(DATASET_PATH)
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    return df.set_index('datetime')

def plot_analysis(df_sim, df_gt=None):
    # Ensure datetime
    df_sim['datetime'] = pd.to_datetime(df_sim['datetime'])
    df_sim = df_sim.set_index('datetime')
    
    # ---------------------------------------------------------
    # 1. FORECAST ACCURACY
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 6))
    
    # Ground Truth (Real)
    # We need to slice GT to match Sim period
    if df_gt is not None:
        start = df_sim.index.min()
        end = df_sim.index.max()
        mask = (df_gt.index >= start) & (df_gt.index <= end)
        df_gt_slice = df_gt[mask]
        
        # Resample GT to match Sim (5min) vs Hourly? 
        # GT is Hourly. Sim is 5min.
        # We plot GT as black line.
        plt.plot(df_gt_slice.index, df_gt_slice['occupancy'] * 1.8, label="Real (Dataset)", color="black", linewidth=2)
    
    # Forecast (from Sim Log which we saved)
    if 'forecast_raw' in df_sim.columns:
        # Check scale
        y_forc = df_sim['forecast_raw']
        if y_forc.max() <= 1.05: # Ratio
            y_forc = y_forc * 180
        elif y_forc.max() <= 105: # %
             y_forc = y_forc * 1.8
        
        plt.plot(df_sim.index, y_forc, label="Forecast (Chronos)", color="green", linestyle="--", alpha=0.8)
    
    # Classic Forecast? Not in CSV unless we saved both.
    # Usually we compare "Forecast used" vs "Real". 
    # In Battle Mode, AI uses Chronos. Classic uses Chronos (in fair mode).
    # So we only have one forecast.
    
    plt.title("Forecast Accuracy (Real vs Predicted)")
    plt.ylabel("Occupancy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, "plot_1_forecast.png"))
    print("[Plot] Saved plot_1_forecast.png")


    # CHECK FOR GRAND BATTLE (4 Contenders)
    if 'Classic_Stat_occ' in df_sim.columns:
        print_grand_battle_report(df_sim)
        return
        
    # ---------------------------------------------------------
    # 2. PRICE STRATEGY
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 6))
    if 'Classic_price' in df_sim.columns:
        plt.plot(df_sim.index, df_sim['Classic_price'], label="Classic", color="gray", alpha=0.7)
    if 'AI_price' in df_sim.columns:
        plt.plot(df_sim.index, df_sim['AI_price'], label="AI (MCTS)", color="blue", linewidth=1.5)
        
    plt.title("Price Strategy Battle (5min Resolution)")
    plt.ylabel("Price (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, "plot_2_price.png"))
    print("[Plot] Saved plot_2_price.png")

    # ---------------------------------------------------------
    # 3. OCCUPANCY RESULT
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 6))
    if 'Classic_occ' in df_sim.columns:
        plt.plot(df_sim.index, df_sim['Classic_occ'], label="Classic", color="gray", alpha=0.6)
    if 'AI_occ' in df_sim.columns:
        plt.plot(df_sim.index, df_sim['AI_occ'], label="AI (MCTS)", color="blue", linewidth=1.5)
        
    plt.title("Occupancy Result (Target 80% = 144)")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, "plot_3_occupancy.png"))
    print("[Plot] Saved plot_3_occupancy.png")
    
    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'CLASSIC':<12} | {'AI':<12} | {'DELTA':<8}")
    print("-" * 80)
    
    # Revenue
    rev_c = (df_sim['Classic_price'] * df_sim['Classic_occ'] * (5/60)).sum()
    rev_ai = (df_sim['AI_price'] * df_sim['AI_occ'] * (5/60)).sum()
    
    # Avg Price
    p_c = df_sim['Classic_price'].mean()
    p_ai = df_sim['AI_price'].mean()
    
    # Avg Occ
    o_c = (df_sim['Classic_occ'] / 180).mean() * 100
    o_ai = (df_sim['AI_occ'] / 180).mean() * 100
    
    print(f"{'Revenue (€)':<20} | {rev_c:12.0f} | {rev_ai:12.0f} | {rev_ai - rev_c:+8.0f}")
    print(f"{'Avg Price (€)':<20} | {p_c:12.2f} | {p_ai:12.2f} | {p_ai - p_c:+8.2f}")
    print(f"{'Avg Occ (%)':<20} | {o_c:12.1f} | {o_ai:12.1f} | {o_ai - o_c:+8.1f}")
    print("="*80)

def print_grand_battle_report(df):
    """
    Calculates metrics for:
    - Classic + Statistical
    - Classic + Chronos
    - MCTS + Statistical
    - MCTS + Chronos
    Prints a Markdown Table.
    """
    contenders = [
        ("Classic", "Statistical", "Classic_Stat"),
        ("Classic", "Chronos",     "Classic_Chronos"),
        ("MCTS",    "Statistical", "MCTS_Stat"),
        ("MCTS",    "Chronos",     "MCTS_Chronos"),
    ]
    
    # Store metrics rows
    rows = []
    
    for pol, fore, key in contenders:
        if f"{key}_price" not in df.columns: continue
        
        # Calculate
        price_col = df[f"{key}_price"]
        occ_col   = df[f"{key}_occ"] # Is usually unscaled (0-180) in CSV export? No, runner exports *180.
        # Wait, runner.py exports: df_export[name_occ] = df_l["occ"] * 180.
        # So it is Vehicles.
        
        # Revenue = Price * Vehicles * TimeStep(h)
        rev = (price_col * occ_col * (5/60)).sum()
        
        avg_price = price_col.mean()
        avg_occ_pct = (occ_col.mean() / 180) * 100
        
        rows.append(f"| {pol:<8} | {fore:<11} | €{rev:,.0f} | €{avg_price:.2f} | {avg_occ_pct:.1f}% |")
        
    print("\n# Grand Battle Report (30 Days)\n")
    print("| Policy   | Forecast    | Revenue  | Avg Price | Occupancy |")
    print("| :---     | :---        | :---     | :---      | :---      |")
    for r in rows:
        print(r)
    print("\n")

def main():
    target_csv = DEFAULT_FILE
    if len(sys.argv) > 1:
        target_csv = sys.argv[1]
    
    print(f"Loading Sim Data: {target_csv}")
    df_sim = load_data(target_csv)
    if df_sim is None: return
    
    print(f"Loading Ground Truth: {DATASET_PATH}")
    df_gt = load_ground_truth()
    
    print("Generating Plots...")
    plot_analysis(df_sim, df_gt)

if __name__ == "__main__":
    main()
