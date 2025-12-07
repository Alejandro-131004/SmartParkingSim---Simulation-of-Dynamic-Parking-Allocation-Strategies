import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from model import Stats


# ============================================================
# LOT
# ============================================================

@dataclass
class TwinLot:
    name: str
    capacity: int
    price: float
    distance: float = 0.0

    def build(self, env):
        # Use a dummy resource structure that allows direct 'count' manipulation
        # This mocks simpy.Resource interface needed by policies (count, capacity)
        class DummyResource:
            def __init__(self, capacity):
                self.capacity = capacity
                self.count = 0  # Can be float
                self.users = [] # Compatibility if checked
        
        self.res = DummyResource(self.capacity)
        self.last = 0
        self.occ_time = 0.0

    def occ_update(self, env):
        dt = env.now - self.last
        if dt > 0:
            self.occ_time += dt * self.res.count
        self.last = env.now


# ============================================================
# STATS
# ============================================================

class TwinStats(Stats):
    """Reuse full Stats structure (revenue, avg_price, occ_lots, etc.) for offline twin."""
    pass


# ============================================================
# OFFLINE DIGITAL TWIN SIMULATOR
# ============================================================

class OfflineDigitalTwinSimulator:

    def __init__(self, cfg):
        self.cfg = cfg

    # ----------------------------------------------------------
    # FORECAST OCC → ARRIVALS
    # ----------------------------------------------------------
    def occupancy_to_arrivals(self, df_pred):
        """
        Convert forecasted occupancy into approximate arrivals (entries/exits)
        while building a valid timeline for the offline twin.
        """

        df = df_pred.copy()

        # Ensure correct columns
        required_cols = ["date", "hour", "forecast_occ"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column in occupancy forecast: {c}")

        # Sort by date/hour
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "hour"]).reset_index(drop=True)

        # --------------------------------------------
        # Reconstruct arrivals (entries) and exits
        # --------------------------------------------
        # Convert occupancy forecast to int
        df["occ"] = df["forecast_occ"].round().astype(int)

        # Compute difference from previous hour
        diff = df["occ"].diff().fillna(0)

        # Positive diff → entries
        df["entries"] = diff.clip(lower=0).astype(int)

        # Negative diff → exits
        df["exits"] = (-diff.clip(upper=0)).astype(int)

        # --------------------------------------------
        # Build a proper timestamp for the simulation
        # --------------------------------------------
        df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

        # Absolute minutes from start (required by the twin)
        df["abs_min"] = (
            (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 60
        ).astype(int)

        # Rename for compatibility with runner
        df.rename(columns={"occ": "occupancy_forecast"}, inplace=True)

        # Keep only final relevant columns
        df = df[
            [
                "date",
                "hour",
                "parking_lot",
                "entries",
                "exits",
                "occupancy_forecast",
                "timestamp",
                "abs_min",
            ]
        ]

        return df

    # ----------------------------------------------------------
    # PROCESS ARRIVALS
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # SIMULATE OCCUPANCY FLOW
    # ----------------------------------------------------------
    def simulate_occupancy_flow(self, env, lot, stats, df_arrivals, elasticity_cfg, base_price, record_price_stats=False):
        """
        Directly modulates occupancy based on price and forecasts.
        Calculates revenue as Integral(Occupancy * Price) dt.
        """
        df_arrivals = df_arrivals.sort_values("abs_min").reset_index(drop=True)
        current_index = 0
        n = len(df_arrivals)
        
        # Elasticity coefficient (default 1.0 if not specified)
        alpha = float(elasticity_cfg.get("price", 1.0))
        
        # We assume 1-minute steps for the simulation loop
        # We iterate through the dataframe which has data per minute (or gaps)
        # But wait, df_arrivals from occupancy_to_arrivals has rows whenever occupancy changes?
        # NO, occupancy_to_arrivals (which we should double check) does diffs.
        # Actually, let's check occupancy_to_arrivals. It returns a dataframe with columns including 'occupancy_forecast'.
        # We should probably iterate minute-by-minute to be safe and accurate with pricing policy updates.
        
        # Let's rebuild the time loop more robustly.
        max_min = int(df_arrivals["abs_min"].max())
        
        # Create a lookup for forecast occupancy
        # df_arrivals has 'abs_min' and 'occupancy_forecast'.
        # We can reindex it to be per-minute if it isn't, but let's assume valid rows.
        # Actually, occupancy_to_arrivals creates rows derived from `df["occ"].diff()`.
        # It might NOT have a row for every minute if occupancy doesn't change.
        # We need the CONTINUOUS forecast signal.
        
        # Let's create a dense lookup
        forecast_lookup = {}
        for _, row in df_arrivals.iterrows():
            t = int(row["abs_min"])
            occ = float(row["occupancy_forecast"])
            forecast_lookup[t] = occ
            
        # We need to fill forward the forecast?
        # No, the source 'occupancy_to_arrivals' comes from 'forecast_occ' which is hourly?
        # Let's look at `runner.py`:
        # df_test["forecast_occ"] ... this is hourly.
        # `occupancy_to_arrivals` converts that to 'abs_min'.
        # It creates a row per HOUR.
        # So we need to interpolate or just step hold? Step hold is fine for hourly forecast.
        
        # Better approach: Iterate from min_t to max_t
        t_start = int(df_arrivals["abs_min"].min())
        t_end = int(df_arrivals["abs_min"].max())
        
        current_forecast = 0.0
        
        for t in range(t_start, t_end + 60): # Run a bit past end
            # Update forecast if new hour
            if t in forecast_lookup:
                current_forecast = forecast_lookup[t]
            
            # 1. Calculate Price Factor
            # factor = 1 - alpha * ( (P - P_base) / P_base )
            if base_price > 1e-9:
                price_ratio = (lot.price - base_price) / base_price
                factor = 1.0 - alpha * price_ratio
            else:
                factor = 1.0
            
            # Constraints
            factor = max(0.0, min(2.0, factor))
            
            # 2. Modulate Occupancy
            target_occ = current_forecast * factor
            
            # 3. Update Lot State
            lot.res.count = target_occ

            # 4. Integrate Revenue
            # Revenue per minute = Occupancy * PricePerHour * (1/60)
            revenue_step = target_occ * lot.price * (1.0 / 60.0)
            stats.revenue += revenue_step
            stats.rev_park += revenue_step # All revenue is parking revenue in this model
            
            # Update Stats Occ Time (used for average occupancy calculations)
            lot.occ_update(env)
            
            # Update Price Stats (every minute)
            # To avoid conflict with Policy (which updates every 5 mins), only do this if requested
            if record_price_stats:
                stats.price_sum[lot.name] += lot.price
                stats.price_n[lot.name] += 1
            
            # Step simulation
            yield env.timeout(1)

    # ----------------------------------------------------------
    # SAMPLER
    # ----------------------------------------------------------
    def sampler(self, env, lot, stats):
        while True:
            row = {
                "t": env.now,
                f"occ_{lot.name}": lot.res.count / lot.capacity,
                f"price_{lot.name}": lot.price,
                f"evutil_{lot.name}": 0.0,  # não há EVs no twin simples
            }
            stats.ts.append(row)
            yield env.timeout(5)


    # ----------------------------------------------------------
    # RUN SIMULATION
    # ----------------------------------------------------------
    def run_simulation(self, df_arrivals, policy=None):

        env = simpy.Environment()
        stats = TwinStats()  # uses Stats structure from model.py

        # Load lot configuration (capacity, price, name, distance)
        lot_cfg_raw = next(l for l in self.cfg["lots"] if l["name"] == "P023")
        valid_keys = ["name", "capacity", "price", "distance"]
        lot_cfg = {k: lot_cfg_raw[k] for k in valid_keys if k in lot_cfg_raw}

        # Build the simulated TwinLot
        lot = TwinLot(**lot_cfg)
        lot.build(env)

        # Initialize price tracking fields for compatibility with DynamicPricingBalanced
        # (Stats has price_sum and price_n dicts for each lot)
        stats.price_sum[lot.name] = 0.0
        stats.price_n[lot.name] = 0

        # Attach dynamic pricing schedule if policy supports it
        if policy is not None and hasattr(policy, "schedule"):
            env.process(policy.schedule(env, [lot], stats))

        # Get elasticity and base price for demand modulation
        elasticity_cfg = self.cfg.get("elasticity", {})
        base_price = lot.price 

        # Process arrivals reconstructed from occupancy forecast
        record_price = (policy is None)
        env.process(self.simulate_occupancy_flow(env, lot, stats, df_arrivals, elasticity_cfg, base_price, record_price_stats=record_price))

        # Sampling timeline every 5 minutes: occupancy, price, EV utilization
        env.process(self.sampler(env, lot, stats))

        # Total simulation time: last absolute minute + buffer
        T = int(df_arrivals["abs_min"].max() + 60)

        # Run the simulation
        env.run(until=T)

        # Return results using Stats.to_dict (expects list of lots)
        return stats.to_dict([lot], T), df_arrivals

