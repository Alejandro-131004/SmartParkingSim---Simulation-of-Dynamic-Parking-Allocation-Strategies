import yaml, json, statistics
import os
import pandas as pd
import numpy as np
from model import run_once
from policies import make_policy
from digital_twin_sim.occupancy_forecast import OccupancyForecaster
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
        print("\n[MODE] Offline Digital Twin Activated\n")

        # -------------------
        # Load and preprocess dataset
        # -------------------
        df = pd.read_csv(cfg["dataset_path"])
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])

        train_start = pd.to_datetime(cfg["twin_train_start"])
        train_end   = pd.to_datetime(cfg["twin_train_end"])
        test_start  = pd.to_datetime(cfg["twin_test_start"])
        test_end    = pd.to_datetime(cfg["twin_test_end"])

        print(f"Training window: {train_start.date()} → {train_end.date()}")
        print(f"Testing window : {test_start.date()} → {test_end.date()}\n")

        df_train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        df_test  = df[(df["date"] >= test_start)  & (df["date"] <= test_end)]

        # ============================================================
        # STEP 1 — Train forecasting model
        # ============================================================
        print("[1/3] Training forecasting model...")
        forecaster = OccupancyForecaster()
        forecaster.fit(df_train)

        # ============================================================
        # STEP 2 — Generate predictions
        # ============================================================
        print("[2/3] Generating forecast for test window...")
        df_pred = forecaster.predict(df_test)

        # Evaluate forecast accuracy
        real = df_pred["occupancy"].values
        pred = df_pred["forecast_occ"].values

        mae = float(abs(pred - real).mean())
        rmse = float(((pred - real) ** 2).mean() ** 0.5)
        real_safe = np.clip(real, a_min=1, a_max=None)
        mape = float((abs(pred - real) / real_safe).mean() * 100)


        print("\n=== FORECAST METRICS ===")
        print(f"MAE : {mae}")
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape} %\n")

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

    # Attach elasticities and event calendar
    if hasattr(policy, "set_elasticities"):
        policy.set_elasticities(cfg.get("elasticity", {}))
    if hasattr(policy, "set_event_calendar"):
        policy.set_event_calendar(cfg.get("event_calendar", []))

    # Run multiple simulations
    results = []
    for r in range(cfg["runs"]):
        seed = cfg["seed"] + r
        out = run_once(cfg, policy, seed)
        out["seed"] = seed
        results.append(out)

    # Save base results
    pd.DataFrame(results).to_csv("reports/results.csv", index=False)

    print("Saved normal simulation results.\n")


if __name__ == "__main__":
    main()
