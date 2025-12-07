import yaml, json, statistics
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from tqdm import tqdm

from model import run_once
from policies import make_policy

from digital_twin_real.forecasting import Forecaster
from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)


def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def main():

    cfg = yaml.safe_load(open("config/parking_P023.yaml"))
    print("EV share type:", type(cfg.get("ev_share")), "value:", cfg.get("ev_share"))

    print("\n================ Execution Mode ================")
    print("1 - Normal Simulation (SimPy)")
    print("2 - Digital Twin (Legacy - deprecated)")
    print("3 - Offline Digital Twin (Forecast + SimPy Simulation + Evaluation)")
    choice = input("\nSelect execution mode (1, 2, or 3): ").strip()

    if choice == "3":
        print("\n[MODE] Offline Digital Twin Activated (Chronos Enabled)\n")

        df = pd.read_csv(cfg["dataset_path"])
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        if "hour" in df.columns:
            df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
        else:
            raise ValueError("Dataset missing 'hour' column")

        test_start = pd.to_datetime(cfg["twin_test_start"])
        test_end   = pd.to_datetime(cfg["twin_test_end"])

        print(f"Testing window : {test_start.date()} -> {test_end.date()}\n")

        mask_test = (df["date"] >= test_start) & (df["date"] <= test_end)
        df_test = df[mask_test].copy().sort_values("timestamp")

        print("[1/3] Loading Amazon Chronos forecasting model...")
        forecaster = Forecaster(cfg["dataset_path"])

        print(f"[2/3] Generating rolling forecast for {len(df_test)} hours...")

        forecasts = []
        full_history = forecaster.df.sort_values("timestamp").reset_index(drop=True)

        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            current_time = row["timestamp"]
            history_context = full_history[full_history["timestamp"] < current_time]
            forecaster.df = history_context
            try:
                pred_occ = forecaster.predict_next_hours(horizon=1)[0]
            except Exception as e:
                print(f"Warning: Forecast failed at {current_time}, using 0. Error: {e}")
                pred_occ = 0.0
            forecasts.append(pred_occ)

        df_test["forecast_occ"] = forecasts
        df_pred = df_test

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

        print("\n[3/3] Running Offline Digital Twin...")

        simulator = OfflineDigitalTwinSimulator(cfg)
        df_arrivals = simulator.occupancy_to_arrivals(df_pred)

        # --- Run 1: Static Baseline (No Policy, fixed price) ---
        print(" -> Running Static Baseline...")
        sim_results_static, _ = simulator.run_simulation(df_arrivals, policy=None)

        # --- Run 2: Dynamic (With Policy) ---
        print(" -> Running Dynamic Model...")
        policy = make_policy(cfg["policy"])
        # Inject config parameters into policy
        for attr in ["target", "k", "p_min", "p_max", "interval"]:
            if attr in cfg and hasattr(policy, attr):
                setattr(policy, attr, float(cfg[attr]) if attr != "interval" else int(cfg[attr]))
        if hasattr(policy, "set_elasticities"):
            policy.set_elasticities(cfg.get("elasticity", {}))
        if hasattr(policy, "set_event_calendar"):
            policy.set_event_calendar(cfg.get("event_calendar", []))

        sim_results, df_arrivals_out = simulator.run_simulation(df_arrivals, policy)

        # --- Comparison Output ---
        def get_metrics(res):
            rev = res.get("revenue_total", 0.0)
            avg_p = float(list(res.get("avg_price", {}).values())[0]) if res.get("avg_price") else 0.0
            occ = res.get("occ_global", 0.0)
            return rev, avg_p, occ

        rev_s, price_s, occ_s = get_metrics(sim_results_static)
        rev_d, price_d, occ_d = get_metrics(sim_results)

        print("\n" + "="*50)
        print(f"{'METRIC':<20} | {'STATIC':<10} | {'DYNAMIC':<10} | {'DELTA':<10}")
        print("-" * 50)
        print(f"{'Revenue (€)':<20} | {rev_s:10.2f} | {rev_d:10.2f} | {rev_d - rev_s:+10.2f}")
        print(f"{'Avg Price (€)':<20} | {price_s:10.2f} | {price_d:10.2f} | {price_d - price_s:+10.2f}")
        print(f"{'Avg Occupancy (%)':<20} | {occ_s*100:10.1f} | {occ_d*100:10.1f} | {(occ_d - occ_s)*100:+10.1f}")
        print("="*50 + "\n")

        analysis = {
            "forecast_metrics": {
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            },
            # forecast_vs_real removed (no longer used)
            "forecast_vs_real": [],
            "simulation_results": sim_results,
            "forecast_arrivals": df_arrivals_out.to_dict(orient="records"),
        }

        output_path = "reports/offline_digital_twin.json"
        with open(output_path, "w") as f:
            json.dump(convert_timestamps(analysis), f, indent=2)

        print(f"\nSaved:\n - {output_path}")
        print(" - reports/offline_digital_twin.json\n")
        return

    if choice == "2":
        print("\nDigital Twin Legacy mode disabled for this version.\n")
        return

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
