import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from model import Stats

# ============================================================
# LOT DEFINITION
# ============================================================

@dataclass
class TwinLot:
    name: str
    capacity: int
    price: float
    distance: float = 0.0

    def build(self, env):
        # Dummy resource to maintain compatibility with policies.py
        class DummyResource:
            def __init__(self, capacity):
                self.capacity = capacity
                self.count = 0 
                self.users = [] 
        
        self.res = DummyResource(self.capacity)
        self.last = 0
        self.occ_time = 0.0

    def occ_update(self, env):
        dt = env.now - self.last
        if dt > 0:
            self.occ_time += dt * self.res.count
        self.last = env.now


# ============================================================
# OFFLINE DIGITAL TWIN SIMULATOR
# ============================================================

class OfflineDigitalTwinSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def sampler(self, env, lot, stats, interval=5):
        """Records metrics periodically."""
        while True:
            # Record Occupancy
            stats.occ_log[lot.name].append({
                "t": env.now,
                "occ": lot.res.count / lot.capacity,
                "price": lot.price
            })
            yield env.timeout(interval)

    def simulate_demand_dynamics(self, env, lot, stats, df_forecast, elasticity_cfg, base_price):
        """
        The core loop of the Digital Twin.
        """
        elast_price = elasticity_cfg.get("price", 1.2) 
        start_time = df_forecast.index[0]
        
        for ts, row in df_forecast.iterrows():
            sim_time_target = (ts - start_time).total_seconds() / 60.0
            delay = sim_time_target - env.now
            
            if delay > 0:
                yield env.timeout(delay)

            lot.occ_update(env)

            base_demand_occ = row['occupancy_pred']
            current_price = lot.price
            
            if base_price > 0 and base_demand_occ > 0:
                price_ratio = current_price / base_price
                price_ratio = max(0.2, min(5.0, price_ratio))
                demand_factor = price_ratio ** (-elast_price)
                target_occupancy_abs = base_demand_occ * demand_factor
            else:
                target_occupancy_abs = base_demand_occ

            target_occupancy_abs = min(target_occupancy_abs, lot.capacity)
            lot.res.count = target_occupancy_abs

    def run(self, policy_instance):
        print(f"[Twin] Initializing Offline Digital Twin...")
        
        env = simpy.Environment()
        stats = Stats() 

        # --- FIX: Initialize custom logs explicitly ---
        # The base Stats class might not have these attributes initialized
        if not hasattr(stats, "occ_log"):
            stats.occ_log = {}
        if not hasattr(stats, "price_sum"):
            stats.price_sum = {}
        if not hasattr(stats, "price_n"):
            stats.price_n = {}
        
        # Setup Lot
        lot_cfg_raw = self.cfg["lots"][0] 
        lot = TwinLot(
            name=lot_cfg_raw["name"], 
            capacity=lot_cfg_raw["capacity"], 
            price=lot_cfg_raw["price"],
            distance=lot_cfg_raw.get("distance", 0)
        )
        lot.build(env)
        
        # Initialize Stats containers for this lot
        stats.occ_log[lot.name] = []
        stats.price_sum[lot.name] = 0.0
        stats.price_n[lot.name] = 0

        # Setup Policy
        policy = policy_instance # It is already an object instance
        
        # Inject params from config into policy
        for k, v in self.cfg.items():
            if k not in ["lots", "policy", "elasticity"]:
                setattr(policy, k, v)
        
        # Start Policy Process if it has a schedule
        if hasattr(policy, "schedule"):
            env.process(policy.schedule(env, [lot], stats))

        # Import Forecast
        try:
            from occupancy_forecast import OccupancyForecaster
        except ImportError:
            from digital_twin_sim.occupancy_forecast import OccupancyForecaster
        
        train_start = self.cfg.get("twin_train_start", "2022-01-01")
        train_end   = self.cfg.get("twin_train_end", "2022-05-31")
        test_start  = self.cfg.get("twin_test_start", "2022-06-01")
        test_end    = self.cfg.get("twin_test_end", "2022-06-30")
        
        print(f"[Twin] Training Forecast Model ({train_start} to {train_end})...")
        df_full = pd.read_csv(self.cfg["dataset_path"])
        
        df_full['date_obj'] = pd.to_datetime(df_full['date'])
        
        mask_train = (df_full['date_obj'] >= pd.to_datetime(train_start)) & \
                     (df_full['date_obj'] <= pd.to_datetime(train_end))
        df_train = df_full[mask_train]
        
        forecaster = OccupancyForecaster()
        forecaster.fit(df_train)
        
        print(f"[Twin] Generating Baseline Demand for Test Period ({test_start} to {test_end})...")
        df_forecast = forecaster.predict(test_start, test_end, stochastic=True)
        
        # Start Simulation
        base_price = lot.price 
        elasticity_cfg = self.cfg.get("elasticity", {})
        
        env.process(self.simulate_demand_dynamics(
            env, lot, stats, df_forecast, elasticity_cfg, base_price
        ))
        
        env.process(self.sampler(env, lot, stats, interval=5))

        t_start = pd.to_datetime(test_start)
        t_end = pd.to_datetime(test_end)
        duration_minutes = (t_end - t_start).total_seconds() / 60
        
        env.run(until=duration_minutes)
        
        print("[Twin] Simulation Complete.")
        
        return stats.occ_log[lot.name], df_forecast