import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from model import Stats

@dataclass
class TwinLot:
    name: str
    capacity: int
    price: float
    distance: float = 0.0

    def build(self, env):
        class DummyResource:
            def __init__(self, capacity):
                self.capacity = capacity
                self.count = 0 
                self.users = [] 
        
        self.res = DummyResource(self.capacity)
        self.last = 0
        self.occ_time = 0.0
        
        # Variáveis de Estado de Fluxo
        self.current_base_entries = 0.0
        self.current_actual_entries = 0.0
        # Novo: Acumulador de carros que queriam sair mas ficaram presos no trânsito
        self.delayed_exits_queue = 0.0

    def occ_update(self, env):
        dt = env.now - self.last
        if dt > 0:
            self.occ_time += dt * self.res.count
        self.last = env.now


class OfflineDigitalTwinSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.traffic_avg_ref = 1.0 

    def sampler(self, env, lot, stats, df_forecast, interval=5):
        """Records metrics including Traffic, Flow Rejection and Congestion."""
        while True:
            try:
                sim_min = int(env.now)
                idx = min(sim_min // 5, len(df_forecast)-1)
                traffic_val = df_forecast.iloc[idx]['traffic_pred']
            except:
                traffic_val = 0.0

            stats.occ_log[lot.name].append({
                "t": env.now,
                "occ": lot.res.count / lot.capacity,
                "price": lot.price,
                "traffic": traffic_val,
                "base_entries": lot.current_base_entries,
                "actual_entries": lot.current_actual_entries,
                "delayed_exits": lot.delayed_exits_queue # Nova Métrica
            })
            yield env.timeout(interval)


    def simulate_flow_dynamics(self, env, lot, stats, df_forecast, elasticity_cfg, base_price, events_cfg=None):
        """
        Full Parking-Flow Simulation including:
        1. Entry Elasticity (Price + Traffic)
        2. Exit Friction (Traffic-Induced Blocking) — Task 4.4.4.2
        3. Event-Based Demand Uplift — Instructor Requirement
        """
        if events_cfg is None:
            events_cfg = []

        # Elasticity parameters
        elast_price = elasticity_cfg.get("price", 1.2)
        elast_traffic = elasticity_cfg.get("traffic", 0.3)

        # Exit friction: how strongly traffic reduces exit throughput
        # Example: 0.5 means that if traffic doubles, exit speed becomes 1/sqrt(2)
        exit_friction = elasticity_cfg.get("exit_friction", 0.5)

        # Initialize simulation state
        start_time = df_forecast.index[0]
        
        # FIX: Get the first row directly from the dataframe to initialize
        first_row = df_forecast.iloc[0]
        real_occ = first_row.get("occupancy_real", None)
        
        if real_occ is None:
            real_occ = first_row["occupancy_pred"]

        lot.res.count = max(0.0, min(real_occ, lot.capacity))

        # Main simulation loop
        for ts, row in df_forecast.iterrows():

            # Synchronize SimPy time with forecast timestamp
            sim_time_target = (ts - start_time).total_seconds() / 60.0
            delay = sim_time_target - env.now
            if delay > 0:
                yield env.timeout(delay)

            lot.occ_update(env)

            # ------------------------------------------------------
            # 0. CHECK FOR EVENT-BASED DEMAND UPLIFT
            # ------------------------------------------------------
            event_factor = 1.0
            # 'day' in config usually refers to day_of_year (1-365)
            current_day = ts.dayofyear
            current_hour = ts.hour + ts.minute / 60.0

            for event in events_cfg:
                # Check if event is active today
                if event.get("day") == current_day:
                    if event.get("start_hour") <= current_hour < event.get("end_hour"):
                        uplift = event.get("demand_uplift", 1.0)
                        event_factor *= uplift
                        
                        # Debug Print for events
                        print(f"[DEBUG EVENT] Day {current_day} Hour {current_hour:.2f}: Applying uplift {uplift}")

            # ------------------------------------------------------
            # 1. READ BASE DEMAND AND TRAFFIC CONDITIONS
            # ------------------------------------------------------
            scale = self.cfg.get("flow_scale", 6.0)   # default: 6x more flow for 5-min from 30-min base
            base_entries = row["entries_pred"] * scale * event_factor
            base_exits   = row["exits_pred"] * scale

            current_traffic = row["traffic_pred"]

            # ------------------------------------------------------
            # 2. COMPUTE ENTRIES (Probability of Entry)
            # ------------------------------------------------------

            # Price elasticity
            price_factor = 1.0
            if base_price > 0 and base_entries > 0:
                price_ratio = max(0.2, min(5.0, lot.price / base_price))
                price_factor = price_ratio ** (-elast_price)

            # Traffic elasticity for entry flow
            traffic_factor_in = 1.0
            if self.traffic_avg_ref > 0 and current_traffic > 0:
                traffic_ratio = current_traffic / self.traffic_avg_ref
                traffic_ratio = max(0.5, min(2.0, traffic_ratio))
                traffic_factor_in = traffic_ratio ** (-elast_traffic)

            actual_entries = base_entries * price_factor * traffic_factor_in

            # ------------------------------------------------------
            # 3. COMPUTE EXITS (Traffic-Constrained Throughput)
            # ------------------------------------------------------
            # Total cars wanting to exit now (new + previously blocked)
            total_pending_exits = base_exits + lot.delayed_exits_queue

            # Exit throughput reduction due to traffic
            exit_throughput_factor = 1.0
            if self.traffic_avg_ref > 0 and current_traffic > 0:
                if current_traffic > self.traffic_avg_ref:
                    t_ratio = current_traffic / self.traffic_avg_ref
                    exit_throughput_factor = 1.0 / (t_ratio ** exit_friction)

            # Realizable number of exits given traffic blockages
            possible_exits = total_pending_exits * exit_throughput_factor

            # Update exit queue (blocked cars)
            lot.delayed_exits_queue = total_pending_exits - possible_exits
            actual_exits = possible_exits

            # Reporting variables
            lot.current_base_entries = base_entries
            lot.current_actual_entries = actual_entries

            # ------------------------------------------------------
            # 4. UPDATE PARKING LOT STATE
            # ------------------------------------------------------
            new_count = lot.res.count + actual_entries - actual_exits
            new_count = max(0.0, min(new_count, lot.capacity))
            lot.res.count = new_count

    

    def run(self, policy_instance):
        import os  # Required for path correction
        print(f"[Twin] Initializing Full Digital Twin (Inputs/Exits Logic)...")
        
        env = simpy.Environment()
        stats = Stats() 

        if not hasattr(stats, "occ_log"): stats.occ_log = {}
        if not hasattr(stats, "price_sum"): stats.price_sum = {}
        if not hasattr(stats, "price_n"): stats.price_n = {}
        
        lot_cfg_raw = self.cfg["lots"][0] 
        lot = TwinLot(name=lot_cfg_raw["name"], capacity=lot_cfg_raw["capacity"], price=lot_cfg_raw["price"])
        lot.build(env)
        
        stats.occ_log[lot.name] = []
        stats.price_sum[lot.name] = 0.0
        stats.price_n[lot.name] = 0

        policy = policy_instance
        # Inject config into policy if needed
        for k, v in self.cfg.items():
            if k not in ["lots", "policy", "elasticity"]: setattr(policy, k, v)
        
        if hasattr(policy, "schedule"):
            env.process(policy.schedule(env, [lot], stats))

        try:
            from digital_twin_sim.occupancy_forecast import OccupancyForecaster
        except ImportError:
            from occupancy_forecast import OccupancyForecaster
        
        train_start = self.cfg.get("twin_train_start", "2022-01-01")
        train_end   = self.cfg.get("twin_train_end", "2022-05-31")
        test_start  = self.cfg.get("twin_test_start", "2022-06-01")
        test_end    = self.cfg.get("twin_test_end", "2022-06-30")
        
        print(f"[Twin] Training Forecast Model...")
        # Check if user is using the LFS pointer file by mistake
        if "ME_2022_S1.csv" in self.cfg["dataset_path"]:
            print("[WARN] It seems you are using the LFS pointer file. Please use 'data/P023_sim/dataset_P023_simready_updated.csv'.")

        # --- PATH CORRECTION LOGIC ---
        dataset_path = self.cfg["dataset_path"]
        if not os.path.exists(dataset_path):
            # Try going up one level (useful when running from src/)
            corrected_path = os.path.join("..", dataset_path)
            if os.path.exists(corrected_path):
                dataset_path = corrected_path
            else:
                print(f"[Error] Dataset not found at '{self.cfg['dataset_path']}' or '{corrected_path}'")
        
        # Load the dataset using the corrected path
        df_full = pd.read_csv(dataset_path)
        # -----------------------------

        df_full['date_obj'] = pd.to_datetime(df_full['date'])
        
        mask_train = (df_full['date_obj'] >= pd.to_datetime(train_start)) & \
                     (df_full['date_obj'] <= pd.to_datetime(train_end))
        df_train = df_full[mask_train]
        
        forecaster = OccupancyForecaster()
        forecaster.fit(df_train)
        self.traffic_avg_ref = forecaster.traffic_base_avg
        
        print(f"[Twin] Generating Forecast...")
        df_forecast = forecaster.predict(test_start, test_end, stochastic=True)

        # Sync real data for comparison
        df_full['datetime'] = pd.to_datetime(df_full['date']) + pd.to_timedelta(df_full['hour'], unit='h')
        df_full = df_full.set_index('datetime').sort_index()

        occ_real = df_full['occupancy'].reindex(df_forecast.index, method='nearest')
        
        # --- PANDAS FIX (Deprecation) ---
        occ_real = occ_real.ffill().bfill()
        # --------------------------------

        df_forecast['occupancy_real'] = occ_real
        
        base_price = lot.price 
        elasticity_cfg = self.cfg.get("elasticity", {})
        
        # Extract event calendar from config
        events_cfg = self.cfg.get("event_calendar", [])

        env.process(self.simulate_flow_dynamics(
            env, lot, stats, df_forecast, elasticity_cfg, base_price, events_cfg
        ))
        
        env.process(self.sampler(env, lot, stats, df_forecast, interval=5))

        t_start = pd.to_datetime(test_start)
        t_end = pd.to_datetime(test_end)
        duration_minutes = (t_end - t_start).total_seconds() / 60
        
        env.run(until=duration_minutes)
        print("[Twin] Simulation Complete.")
        
        return stats.occ_log[lot.name], df_forecast