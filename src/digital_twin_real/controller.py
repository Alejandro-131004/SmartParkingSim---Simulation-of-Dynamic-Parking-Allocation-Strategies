import random
import numpy as np
import simpy

from policies import make_policy
from model import Stats, Lot, _make_sample_duration, sampler

from digital_twin.state_manager import StateManager
from digital_twin.forecasting import Forecaster
from digital_twin.sync import TwinSynchronizer
from digital_twin.model_forecast import arrivals_from_forecast


class DigitalTwinController:
    """
    Full Digital Twin controller:
      - Reads real current state (occupancy, hour, etc.)
      - Computes forecast for the next hours
      - Runs a forecast-driven simulation starting from the real occupancy
        (t=0 = 'current moment' in simulation time)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dataset = cfg["dataset_path"]

        self.state_manager = StateManager(self.dataset)
        self.forecaster = Forecaster(self.dataset)
        self.synchronizer = TwinSynchronizer()

    # -------------------------------------------------------------
    # REAL CURRENT STATE
    # -------------------------------------------------------------
    def get_current_state(self) -> dict:
        return self.state_manager.load_current_state()

    # -------------------------------------------------------------
    # FORECAST
    # -------------------------------------------------------------
    def forecast(self, horizon: int = 3) -> dict:
        occ = self.forecaster.predict_next_hours(horizon)
        arr = self.forecaster.predict_arrivals(horizon)
        return {
            "forecast_occupancy": occ,
            "forecast_arrivals": arr,
        }

    # -------------------------------------------------------------
    # FORECAST-DRIVEN SIMULATION
    # -------------------------------------------------------------
    def run_future_simulation_using_forecast(
        self,
        policy,
        forecast: dict,
        horizon_hours: int,
        run_idx: int = 0,
    ) -> dict:
        """
        Run a simulation of the future starting from the REAL state.
        The simulation uses ONLY forecasted arrivals.
        """

        # --- Seeding (same behavior as run_once) ---
        base_seed = int(self.cfg.get("seed", 42))
        seed = base_seed + run_idx
        random.seed(seed)
        np.random.seed(seed)

        # --- Environment + Stats ---
        env = simpy.Environment()
        stats = Stats()

        # --- Create lots (same structure as run_once) ---
        lots = [Lot(**d) for d in self.cfg["lots"]]
        for L in lots:
            L.build(env)

            stats.arrivals_expected_by_lot[L.name] = 0
            stats.arrivals_spawned_by_lot[L.name] = 0
            stats.served_by_lot[L.name] = 0
            stats.rejected_by_lot[L.name] = 0
            stats.price_sum[L.name] = 0.0
            stats.price_n[L.name] = 0

        # --- Optional dynamic pricing schedule ---
        if hasattr(policy, "schedule"):
            env.process(policy.schedule(env, lots, stats))

        # --- REAL STATE ---
        state = self.get_current_state()
        current_occ = int(state["current_occupancy"])

        # IMPORTANT:
        #   - env.now = 0 (SimPy does not allow modifying it)
        #   - t=0 represents the current real moment
        for lot in lots:
            occ = min(current_occ, lot.capacity)

            # Pre-fill Resource with fake users to represent real occupancy
            for _ in range(occ):
                lot.res.users.append(object())

            lot.last = 0  # occupancy integration starts now

        # --- Duration sampler (same as run_once) ---
        sample_duration = _make_sample_duration(self.cfg)

        # --- Spawn forecast arrivals ---
        future_arrivals = forecast["forecast_arrivals"]

        env.process(
            arrivals_from_forecast(
                env=env,
                lots=lots,
                forecast_arrivals=future_arrivals,
                policy=policy,
                stats=stats,
                sample_duration=sample_duration,
                max_wait=self.cfg.get("max_wait", 1),
                cfg=self.cfg,
            )
        )

        # --- Sampler for timeline ---
        env.process(sampler(env, lots, stats, step=5))

        # --- Run only for the horizon ---
        horizon_minutes = int(horizon_hours * 60)
        env.run(until=horizon_minutes)

        result = stats.to_dict(lots, horizon_minutes)
        result["dt_seed"] = seed
        result["dt_horizon_minutes"] = horizon_minutes
        result["dt_run_id"] = run_idx

        return result

    # -------------------------------------------------------------
    # MULTI-RUN WRAPPER
    # -------------------------------------------------------------
    def simulate_future(
        self,
        policy_name: str = "balanced",
        runs: int = 1,
        forecast: dict | None = None,
        horizon: int = 3,
    ) -> list:
        policy = make_policy(policy_name)
        results = []

        for r in range(runs):
            out = self.run_future_simulation_using_forecast(
                policy=policy,
                forecast=forecast,
                horizon_hours=horizon,
                run_idx=r,
            )
            results.append(out)

        return results

    # -------------------------------------------------------------
    # MAIN DIGITAL TWIN ENTRY POINT
    # -------------------------------------------------------------
    def run_digital_twin(self, horizon: int = 3, runs: int = 1) -> dict:
        """
        Entry point used by runner.py in Digital Twin mode.
        """

        # 1) REAL STATE
        state = self.get_current_state()

        # 2) FORECAST
        forecast = self.forecast(horizon)

        # 3) RUN FUTURE SIMULATION
        simulations = self.simulate_future(
            runs=runs,
            forecast=forecast,
            horizon=horizon,
            policy_name=self.cfg["policy"],
        )

        return {
            "current_state": state,
            "forecast": forecast,
            "simulated_future": simulations,
        }
