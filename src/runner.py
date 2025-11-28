import yaml, json, statistics
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from tqdm import tqdm  # You might need to pip install tqdm for the progress bar

from model import run_once
from policies import make_policy

# CHANGED: Import the new Chronos-based Forecaster from the 'real' folder
from digital_twin_real.forecasting import Forecaster
from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)


# ------------------------------------------------------------
# Helper: make timestamps JSON serializable
# ------------------------------------------------------------
def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


# ============================================================
# MAIN
# ============================================================
def main():

    # Load correct YAML
    cfg = yaml.safe_load(open("config/parking_P023.yaml"))
    print("EV share type:", type(cfg.get("ev_share")), "value:", cfg.get("ev_share"))

    # ============================================================
    # SELECT EXECUTION MODE
    # ============================================================
    print("\n================ Execution Mode ================")
    print("1 - Normal Simulation (SimPy)")
    print("2 - Digital Twin (Legacy - deprecated)")
    print("3 - Offline Digital Twin (Forecast + SimPy Simulation + Evaluation)")
    choice = input("\nSelect execution mode (1, 2, or 3): ").strip()

    # ============================================================
    # MODE 3 — OFFLINE DIGITAL TWIN
    # ============================================================
    if choice == "3":
        print("\n[MODE] Offline Digital Twin Activated (Chronos Enabled)\n")

        # -------------------
        # Load and preprocess dataset
        # -------------------
        # We load it here just to get the test split dates
        df = pd.read_csv(cfg["dataset_path"])
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        # Create a full timestamp column for easier filtering
        if "hour" in df.columns:
            df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
        else:
            raise ValueError("Dataset missing 'hour' column")

        test_start  = pd.to_datetime(cfg["twin_test_start"])
        test_end    = pd.to_datetime(cfg["twin_test_end"])

        print(f"Testing window : {test_start.date()} → {test_end.date()}\n")

        # Create the test slice to iterate over
        mask_test = (df["date"] >= test_start) & (df["date"] <= test_end)
        df_test = df[mask_test].copy().sort_values("timestamp")

        # ============================================================
        # STEP 1 — Initialize Forecasting Model
        # ============================================================
        print("[1/3] Loading Amazon Chronos forecasting model...")
        # Initialize the forecaster with the dataset path so it has access to history
        forecaster = Forecaster(cfg["dataset_path"])
        # Note: No .fit() needed for pre-trained models

        # ============================================================
        # STEP 2 — Generate predictions (Rolling Forecast)
        # ============================================================
        print(f"[2/3] Generating rolling forecast for {len(df_test)} hours...")
        
        forecasts = []
        
        # We need to simulate the passage of time. For every hour in the test set,
        # we ask the model to predict the *next* hour based on data *up to that point*.
        # NOTE: This loop can be slow on CPU.
        
        # Ensure the internal dataframe in forecaster is sorted
        full_history = forecaster.df.sort_values("timestamp").reset_index(drop=True)
        
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            current_time = row["timestamp"]
            
            # Create a context: all data strictly BEFORE this specific hour
            # The Forecaster class needs to support passing custom context, or we filter manually.
            # To keep it simple with your current Forecaster class, we modify the internal df temporarily
            # or rely on the fact that Forecaster uses `self.df`.
            
            # Update the forecaster's internal data to only include history up to now
            # (Simulating that we don't know the future)
            history_context = full_history[full_history["timestamp"] < current_time]
            
            # Inject this restricted history into the forecaster instance
            forecaster.df = history_context
            
            # Predict 1 step ahead (next hour)
            # The 'predict_next_hours' uses the end of forecaster.df as the start point
            try:
                pred_occ = forecaster.predict_next_hours(horizon=1)[0]
            except Exception as e:
                print(f"Warning: Forecast failed at {current_time}, using 0. Error: {e}")
                pred_occ = 0.0
            
            forecasts.append(pred_occ)

        # Assign predictions back to df_test
        df_test["forecast_occ"] = forecasts
        df_pred = df_test  # Rename for compatibility

        # Evaluate forecast accuracy
        real = df_pred["occupancy"].values
        pred = df_pred["forecast_occ"].values

        mae = float(abs(pred - real).mean())
        rmse = float(((pred - real) ** 2).mean() ** 0.5)
        real_safe = np.clip(real, a_min=1, a_max=None)
        mape = float((abs(pred - real) / real_safe).mean() * 100)

        print("\n=== FORECAST METRICS ===")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f} %\n")

        # ============================================================
        # STEP 3 — Simulate using forecasted arrivals
        # ============================================================
        print("[3/3] Running SimPy simulation using forecasted arrivals...")

        simulator = OfflineDigitalTwinSimulator(cfg)
        df_arrivals = simulator.occupancy_to_arrivals(df_pred)
        sim_results, df_arrivals = simulator.run_simulation(df_arrivals, None)

        # ============================================================
        # SAVE OUTPUT
        # ============================================================
        output = {
            "forecast_metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            },
            "forecast_vs_real": df_pred.to_dict(orient="records"),
            "simulation_results": sim_results,
            "forecast_arrivals": df_arrivals.to_dict(orient="records")
        }

        with open("reports/offline_digital_twin.json", "w") as f:
            json.dump(convert_timestamps(output), f, indent=2)

        print("\nSaved:")
        print(" - reports/offline_digital_twin.json\n")
        return


    # ============================================================
    # MODE 2 — Legacy Digital Twin
    # ============================================================
    if choice == "2":
        print("\nDigital Twin Legacy mode disabled for this version.\n")
        return


    # ============================================================
    # MODE 1 — NORMAL SIMULATION (ORIGINAL)
    # ============================================================
    print("\n[MODE] Normal Simulation\n")

    policy = make_policy(cfg["policy"])

    if hasattr(policy, "set_elasticities"):
        policy.set_elasticities(cfg.get("elasticity", {}))
    if hasattr(policy, "set_event_calendar"):
        policy.set_event_calendar(cfg.get("event_calendar", []))

    results = []
    for r in range(cfg["runs"]):
        seed = cfg["seed"] + r
        out = run_once(cfg, policy, seed)
        out["seed"] = seed
        results.append(out)

    pd.DataFrame(results).to_csv("reports/results.csv", index=False)

    print("Saved normal simulation results.\n")


if __name__ == "__main__":
    main()