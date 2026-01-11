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


    def _calculate_sigmoid_elasticity(self, price_ratio, k=2.0, weather_score=0.0):
        """
        Sigmoid Elasticity Function with Weather Adaptation.
        Returns a demand multiplier between 0.0 and 3.0 (weather can boost demand).
        
        Args:
            price_ratio (float): Current Price / Base Price.
            k (float): Steepness.
            weather_score (float): 0.0 (Good) to 1.0 (Bad).
                                   Bad weather reduces price sensitivity and increases max demand.
        """
        # 1. Weather makes people less price sensitive (Willingness to Pay increases)
        # effectively "shrinking" the price ratio.
        # e.g. Price 1.5x feels like 1.15x if score is 1.0
        effective_ratio = price_ratio / (1.0 + 0.3 * weather_score)

        # 2. Main Sigmoid
        val = k * (effective_ratio - 1.0)
        val = max(-100, min(100, val)) 
        base_factor = 2.0 / (1.0 + np.exp(val))

        # 3. Weather-driven Demand Surge (if price is favorable)
        # If price is low/fair, bad weather drives even more people to park (comfort/necessity)
        if effective_ratio < 1.05:
            base_factor *= (1.0 + 0.5 * weather_score) # Max 1.5x boost on top of 2.0x = 3.0x

        return base_factor

    def simulate_flow_dynamics(self, env, lot, stats, df_forecast, elasticity_cfg, base_price, events_cfg=None):
        """
        State-Based Simulation (Occupancy Driven).
        FIXED: Correctly handles scaling for Ratio vs Percentage inputs.
        """
        if events_cfg is None:
            events_cfg = []

        # Elasticity parameters
        elast_price = elasticity_cfg.get("price", 1.2)
        elast_traffic = elasticity_cfg.get("traffic", 0.3)

        # Initialize simulation state
        start_time = df_forecast.index[0]
        
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
            current_day = ts.dayofyear
            current_hour = ts.hour + ts.minute / 60.0

            for event in events_cfg:
                if event.get("day") == current_day:
                    if event.get("start_hour") <= current_hour < event.get("end_hour"):
                        uplift = event.get("demand_uplift", 1.0)
                        event_factor *= uplift

            # ------------------------------------------------------
            # 1. READ BASE DEMAND (Occupancy & Flow)
            # ------------------------------------------------------
            raw_occ = row["occupancy_pred"]
            raw_ent = row["entries_pred"]
            
            # Detect if 'occupancy_pred' is a Ratio (0.0-1.0) or Percentage/Count (>1.5)
            # The Forecaster usually normalizes to Ratio (0.40), so we multiply by Capacity (180) to get cars.
            
            if raw_occ <= 1.5:
                # It is a ratio (e.g., 0.40). Convert to vehicles.
                base_occ = raw_occ * lot.capacity * event_factor
            else:
                # It is likely raw count or percentage.
                # Since we know dataset is %, assume it might not have been normalized?
                # Safer fallback: treat as count, but safeguard.
                base_occ = raw_occ * event_factor
                
            # ENTRIES Logic:
            # The CSV shows entries are raw counts (e.g. 58). 
            # We should NOT multiply entries by capacity. Just apply event/demand multipliers.
            base_entries = raw_ent * event_factor
            # --- CRITICAL FIX END ---
            
            current_traffic = row["traffic_pred"]

            # ------------------------------------------------------
            # 2. COMPUTE ELASTICITY FACTORS
            # ------------------------------------------------------

            # Price elasticity (Sigmoid Model)
            price_factor = 1.0
            
            # Calculate Weather Score (0.0 good -> 1.0 bad) from avg_temp
            avg_temp = row.get("avg_temp", 20.0)
            weather_score = min(1.0, abs(avg_temp - 20.0) / 10.0)

            if base_price > 0:
                price_ratio = lot.price / base_price
                price_factor = self._calculate_sigmoid_elasticity(price_ratio, k=2.0, weather_score=weather_score)

            # Traffic elasticity
            traffic_factor = 1.0
            if self.traffic_avg_ref > 0 and current_traffic > 0:
                traffic_ratio = current_traffic / self.traffic_avg_ref
                traffic_ratio = max(0.5, min(2.0, traffic_ratio))
                traffic_factor = traffic_ratio ** (-elast_traffic)

            # Total modifier
            total_factor = price_factor * traffic_factor

            # ------------------------------------------------------
            # 3. SET PARKING STATE DIRECTLY
            # ------------------------------------------------------
            target_occ = base_occ * total_factor
            
            # Physical constraint
            final_occ = max(0.0, min(target_occ, lot.capacity))
            
            # Update the resource
            if hasattr(lot.res, 'count'):
                 lot.res.count = final_occ
            
            # ------------------------------------------------------
            # 4. UPDATE METRICS
            # ------------------------------------------------------
            lot.current_base_entries = base_entries
            lot.current_actual_entries = base_entries * total_factor
            lot.delayed_exits_queue = 0.0
    

    def run(self, policy_instance, forecaster_type="statistical", precomputed_forecast=None):
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


        try:
            from digital_twin_sim.occupancy_forecast import OccupancyForecaster
            # Conditionally import Chronos
            if forecaster_type == "chronos":
                try:
                    from digital_twin_sim.chronos_adapter import ChronosAdapter
                except ImportError:
                    print("[Error] Could not import ChronosAdapter. Falling back to Statistical.")
                    forecaster_type = "statistical"
        except ImportError:
            from occupancy_forecast import OccupancyForecaster
            if forecaster_type == "chronos":
                try:
                    from chronos_adapter import ChronosAdapter
                except ImportError:
                    forecaster_type = "statistical"
        
        train_start = self.cfg.get("twin_train_start", "2022-01-01")
        train_end   = self.cfg.get("twin_train_end", "2022-05-31")
        test_start  = self.cfg.get("twin_test_start", "2022-06-01")
        test_end    = self.cfg.get("twin_test_end", "2022-06-30")
        
        print(f"[Twin] Training Forecast Model ({forecaster_type.upper()})...")
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
        
        if forecaster_type == "chronos":
            forecaster = ChronosAdapter()
        else:
            forecaster = OccupancyForecaster()
        
        forecaster.fit(df_train)
        self.traffic_avg_ref = forecaster.traffic_base_avg
        
        print(f"[Twin] Generating Forecast...")
        if precomputed_forecast is not None:
             print(f"[Twin] Using Precomputed Forecast ({len(precomputed_forecast)} rows).")
             df_forecast = precomputed_forecast.copy()
        elif forecaster_type == "chronos":
            # For Chronos, we pass the full dataframe (Ground Truth) to enable Rolling Forecast
            # DISABLE for speed (User requested run all tests, rolling is 5 hours)
            # df_forecast = forecaster.predict(test_start, test_end, stochastic=True, ground_truth_df=df_full)
            print("[Twin] Using Static Chronos Forecast (Fast Mode).")
            df_forecast = forecaster.predict(test_start, test_end, stochastic=True, ground_truth_df=None)
        else:
            df_forecast = forecaster.predict(test_start, test_end, stochastic=True)

        # Apply Demand Multiplier (Scenario Testing)
        demand_mult = self.cfg.get("demand_multiplier", 1.0)
        if demand_mult != 1.0:
            # print(f"[Twin] Applying Demand Multiplier: {demand_mult}x")
            df_forecast['occupancy_pred'] *= demand_mult
            df_forecast['entries_pred'] *= demand_mult
            # Ensure capacity constraints (simulation logic handles it, but forecast should be unbounded to show potential)

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

        # Inject config into policy if needed
        for k, v in self.cfg.items():
            if k not in ["lots", "policy", "elasticity"]: setattr(policy, k, v)
        
        if hasattr(policy, "set_forecast"):
            policy.set_forecast(df_forecast)

        if hasattr(policy, "schedule"):
            env.process(policy.schedule(env, [lot], stats))
        
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