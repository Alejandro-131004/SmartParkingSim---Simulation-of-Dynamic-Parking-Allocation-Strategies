import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'digital_twin_sim'))

from src.digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator
from src.policies import make_policy
from src.digital_twin_sim.occupancy_forecast import OccupancyForecaster
try:
    from src.digital_twin_sim.chronos_adapter import ChronosAdapter
except ImportError:
    ChronosAdapter = None

def calculate_forecast_metrics(y_true, y_pred, name):
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    if len(common_idx) == 0:
        return {"MSE": 0, "R2": 0, "MAE": 0}
    
    y_t = y_true.loc[common_idx]
    y_p = y_pred.loc[common_idx]
    
    mse = mean_squared_error(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    return {f"{name}_MSE": mse, f"{name}_R2": r2, f"{name}_MAE": mae}

def run_deep_dive():
    print(">>> Starting Deep Dive Analysis (3x Demand)...")
    
    # Configuration (Hardcoded for 3x Analysis as requested)
    cfg = {
        "lots": [{"name": "P023", "capacity": 180, "price": 1.5}],
        "dataset_path": "data/P023_sim/dataset_P023_simready_updated.csv",
        "twin_train_start": "2022-01-01",
        "twin_train_end": "2022-05-31",
        "twin_test_start": "2022-06-01",
        "twin_test_end": "2022-06-30",
        "demand_multiplier": 3.0,
        "elasticity": {"price": 1.2, "traffic": 0.3},
        "policy": {"type": "dynpricing"} # Placeholder, overridden later
    }
    
    # ---------------------------------------------------------
    # 1. EVALUATE FORECASTS (Classic vs Chronos vs Real)
    # ---------------------------------------------------------
    print("\n[1/4] Evaluating Forecasting Models (Statistical vs All-Star Chronos)...")
    
    # We need to manually load data to get 'Real' ground truth for the test period
    df_full = pd.read_csv(cfg["dataset_path"])
    df_full['date_obj'] = pd.to_datetime(df_full['date'])
    df_full['datetime'] = pd.to_datetime(df_full['date']) + pd.to_timedelta(df_full['hour'], unit='h')
    df_full = df_full.set_index('datetime').sort_index()
    
    mask_test = (df_full['date_obj'] >= pd.to_datetime(cfg["twin_test_start"])) & \
                (df_full['date_obj'] <= pd.to_datetime(cfg["twin_test_end"]))
    df_test_gt = df_full[mask_test].copy()
    
    # Ensure correct scaling (Dataset is 0-100%, we want Cars 0-180 for "Occupancy")
    # Actually forecaster inputs raw data. The Sim does the scaling.
    # User asked for "real ocupation" vs "forecasted".
    # Should we compare in "Dataset Unit" (0-100) or "Capacity Unit" (0-180)?
    # To match Simulation results later, we should probably scale EVERYTHING to Capacity.
    # Raw dataset max is ~66. Capacity is 180.
    # Scale Factor = 1.8.
    scale = 1.8
    y_test_cars = df_test_gt['occupancy'] * scale
    
    # A. Statistical Forecast
    print("   -> Fitting Statistical Model...")
    stat_forecaster = OccupancyForecaster()
    mask_train = (df_full['date_obj'] >= pd.to_datetime(cfg["twin_train_start"])) & \
                 (df_full['date_obj'] <= pd.to_datetime(cfg["twin_train_end"]))
    stat_forecaster.fit(df_full[mask_train])
    df_stat = stat_forecaster.predict(cfg["twin_test_start"], cfg["twin_test_end"], stochastic=False)
    # Apply 3x Multiplier?
    # User said "do all at 3x". 
    # If we compare Forecast accuracy against Real, 3x doesn't make sense (Real is 1x).
    # But for "Forecast used in Sim", it is 3x.
    # Comparison 1: Model Accuracy (1x).
    # Comparison 2: Sim Inputs (3x).
    # User said "diference betwin real ... and chronos ... metrics like mse".
    # This implies 1x (Evaluation of the model).
    # So I will calculate Metrics on 1x.
    # But later use 3x for Sim.
    
    # B. Chronos Forecast
    print("   -> Fitting Chronos Model...")
    df_chronos = None
    if ChronosAdapter:
        chronos_forecaster = ChronosAdapter()
        chronos_forecaster.fit(df_full[mask_train])
        # Rolling forecast with ground truth (DISABLED FOR SPEED - Static Forecast 30 Days)
        # df_chronos = chronos_forecaster.predict(cfg["twin_test_start"], cfg["twin_test_end"], 
        #                                        stochastic=True, ground_truth_df=df_full)
        df_chronos = chronos_forecaster.predict(cfg["twin_test_start"], cfg["twin_test_end"], stochastic=True)
    else:
        print("   -> Chronos not available.")

    # Calculate Metrics (1x)
    print("   -> Calculating Accuracy Metrics...")
    # SCALE FACTOR CORRECTION:
    # 1. Ground Truth (df_test_gt) is 0-100. Scale by 1.8 to get 0-180.
    # 2. Classic Forecast (df_stat) is 0-1 (Ratio, internal norm). Scale by 180.
    # 3. Chronos Forecast (df_chronos) uses Raw Context (0-100). Scale by 1.8.
    
    y_test_cars = df_test_gt['occupancy'] * 1.8
    y_stat = df_stat['occupancy_pred'] * 180.0
    
    met_stat = calculate_forecast_metrics(y_test_cars, y_stat, "ClassicEst")
    
    met_chronos = {}
    y_chronos = None
    if df_chronos is not None:
        y_chronos = df_chronos['occupancy_pred'] * 1.8
        met_chronos = calculate_forecast_metrics(y_test_cars, y_chronos, "Chronos")

    print(f"      Classic: MSE={met_stat['ClassicEst_MSE']:.2f}, R2={met_stat['ClassicEst_R2']:.2f}")
    if met_chronos:
        print(f"      Chronos: MSE={met_chronos['Chronos_MSE']:.2f}, R2={met_chronos['Chronos_R2']:.2f}")

    # PLOT 1: Forecast Comparison (1x or 3x? User wants "difference". I'll plot 1x for clarity of accuracy).
    plt.figure(figsize=(15, 6))
    limit = 24 * 7 * 2 # 1 week (steps) - 5min resolution = 288*7
    # Actually indexes are datetime.
    
    # Resample to hourly for cleaner plot?
    y_real_plot = y_test_cars.resample('H').mean()
    y_stat_plot = y_stat.resample('H').mean()
    if y_chronos is not None: y_chronos_plot = y_chronos.resample('H').mean()
    
    plt.plot(y_real_plot.index, y_real_plot.values, label="Real (Observed)", color="black", linewidth=2)
    plt.plot(y_stat_plot.index, y_stat_plot.values, label="Classic Forecast", linestyle="--", color="gray")
    if y_chronos is not None:
        plt.plot(y_chronos_plot.index, y_chronos_plot.values, label="Chronos (AI)", linestyle="-", color="green", alpha=0.8)
    
    plt.title("Forecast Model Accuracy Comparison (1x Demand)")
    plt.ylabel("Occupancy (Vehicles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/deep_dive_1_forecast_accuracy.png")
    print("   -> Plot Saved: reports/deep_dive_1_forecast_accuracy.png")


    # ---------------------------------------------------------
    # 2. RUN SIMULATIONS (3x Demand)
    # ---------------------------------------------------------
    print("\n[2/4] Running Simulations (Battle Mode @ 3x Demand)...")
    
    # Setup 3x
    cfg["demand_multiplier"] = 3.0
    # Use user-calibrated params from recent file edits (assumed in policies.py defaults or we inject them)
    # User edited defaults in runner logic. I should define robust defaults here.
    # From Step 1431: Target=0.8, k=1.0, p_max=5.0. 
    # Ideally I read from config, but I hardcoded cfg above. 
    # I'll update cfg to match 'calibrated' expectations.
    cfg["policy"]["target_occupancy"] = 0.8
    cfg["policy"]["k"] = 1.0
    cfg["policy"]["p_min"] = 0.0
    cfg["policy"]["p_max"] = 5.0
    
    # A. Classic (Integral)
    print("   -> Simulating Classic (DynamicPricingBalanced)...")
    sim_c = OfflineDigitalTwinSimulator(cfg)
    pol_c = make_policy("dynpricing") 
    # Inject params manually to be safe
    pol_c.target = 0.8
    pol_c.k = 1.0
    pol_c.p_max = 5.0
    
    # OPTIMIZATION: Use precomputed forecast if available to avoid re-running Chronos
    log_c, _ = sim_c.run(pol_c, forecaster_type="chronos", precomputed_forecast=df_chronos)
    df_c = pd.DataFrame(log_c)
    
    # B. AI (MCTS)
    print("   -> Simulating AI (MCTS Agent)...")
    sim_ai = OfflineDigitalTwinSimulator(cfg)
    pol_ai = make_policy("mcts")
    pol_ai.p_max = 5.0
    
    log_ai, _ = sim_ai.run(pol_ai, forecaster_type="chronos", precomputed_forecast=df_chronos)
    df_ai = pd.DataFrame(log_ai)
    
    # Calculate Business Metrics
    def fast_metrics(df, cap=180):
        rev = (df['occ'] * cap * df['price'] * (5/60)).sum()
        avg_occ = df['occ'].mean() * 100
        avg_price = df['price'].mean()
        return rev, avg_occ, avg_price
        
    met_c = fast_metrics(df_c)
    met_ai = fast_metrics(df_ai)


    # ---------------------------------------------------------
    # 3. PLOT PRICE & OCCUPANCY (Comparison)
    # ---------------------------------------------------------
    print("\n[3/4] Generating Comparison Plots...")
    
    # Align time axes
    t_idx = pd.to_datetime(cfg["twin_test_start"]) + pd.to_timedelta(df_c['t'], unit='m')
    df_c['time'] = t_idx
    df_ai['time'] = t_idx
    
    # Slice first 48h or 1 week for visibility
    # User said "price variation... at 3x".
    plot_limit = 24 * 7 * 12 # 1 week = 2016 steps (5min)
    
    # Resample for smooth lines (Hourly)
    df_c_h = df_c.set_index('time').resample('H').mean()
    df_ai_h = df_ai.set_index('time').resample('H').mean()
    
    # PLOT 2: Price Variation (Raw 5-min to show spikes)
    plt.figure(figsize=(15, 6))
    plt.plot(df_c.set_index('time').index, df_c['price'], label="Classic (Integral)", color="gray", linewidth=1.0, alpha=0.7)
    plt.plot(df_ai.set_index('time').index, df_ai['price'], label="AI (MCTS)", color="blue", linewidth=1.5, alpha=0.8)
    plt.title("Price Strategy Battle (3x Demand) - 5min Resolution")
    plt.ylabel("Price (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/deep_dive_2_price_strategy.png")
    print("   -> Plot Saved: reports/deep_dive_2_price_strategy.png")

    # PLOT 3: Occupancy Comparison (Hourly Smooth is okay for Occ, or keep Raw?)
    # User asked about variance, so Raw is better.
    plt.figure(figsize=(15, 6))
    plt.plot(df_c.set_index('time').index, df_c['occ']*180, label="Classic Occ", color="gray", linewidth=1.0, alpha=0.6)
    plt.plot(df_ai.set_index('time').index, df_ai['occ']*180, label="AI Occ", color="blue", linewidth=1.5, alpha=0.8)
    plt.title("Occupancy Result (3x Demand) - 5min Resolution")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/deep_dive_3_occupancy_result.png")
    print("   -> Plot Saved: reports/deep_dive_3_occupancy_result.png")


    # ---------------------------------------------------------
    # 4. FINAL REPORT
    # ---------------------------------------------------------
    print("\n[4/4] Final Metrics Report")
    print("="*80)
    print(f"{'Metric':<20} | {'Classic':<12} | {'MCTS (AI)':<12} | {'Delta':<8}")
    print("-" * 80)
    
    rev_c, occ_c, p_c = met_c
    rev_ai, occ_ai, p_ai = met_ai
    
    # Detailed Stats
    p_min_c, p_max_c = df_c['price'].min(), df_c['price'].max()
    p_min_ai, p_max_ai = df_ai['price'].min(), df_ai['price'].max()
    occ_std_c = df_c['occ'].std() * 100
    occ_std_ai = df_ai['occ'].std() * 100
    
    print(f"{'Revenue (€)':<20} | {rev_c:12.0f} | {rev_ai:12.0f} | {rev_ai - rev_c:+8.0f}")
    print(f"{'Avg Price (€)':<20} | {p_c:12.2f} | {p_ai:12.2f} | {p_ai - p_c:+8.2f}")
    print(f"{'Price Range (€)':<20} | {p_min_c:.1f} - {p_max_c:.1f}  | {p_min_ai:.1f} - {p_max_ai:.1f}  |")
    print(f"{'Avg Occ (%)':<20} | {occ_c:12.1f} | {occ_ai:12.1f} | {occ_ai - occ_c:+8.1f}")
    print(f"{'Occ StdDev (%)':<20} | {occ_std_c:12.1f} | {occ_std_ai:12.1f} | {occ_std_ai - occ_std_c:+8.1f}")
    print("-" * 80)
    print("FORECAST ACCURACY (MSE - Lower is Better):")
    print(f"Classic Forecast: {met_stat['ClassicEst_MSE']:.1f}")
    if met_chronos:
        print(f"Chronos Forecast: {met_chronos['Chronos_MSE']:.1f}")
    print("="*80)

if __name__ == "__main__":
    run_deep_dive()
