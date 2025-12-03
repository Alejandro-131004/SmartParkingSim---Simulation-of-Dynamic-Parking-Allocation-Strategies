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
        self.res = simpy.Resource(env, capacity=self.capacity)
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
    def arrivals_from_forecast(self, env, lot, stats, df_arrivals):
        """
        Generate arrivals and exits per minute from reconstructed forecast arrivals.
        """

        df_arrivals = df_arrivals.sort_values("abs_min").reset_index(drop=True)
        current_index = 0
        n = len(df_arrivals)
        uid_counter = 0   # <<< NOVO: UID para cada carro

        while current_index < n:
            row = df_arrivals.iloc[current_index]

            # Jump time to the correct minute
            t = int(row["abs_min"])
            if t > env.now:
                yield env.timeout(t - env.now)

            # ------------------------------------------------------
            # ENTRADAS (entries)
            # ------------------------------------------------------
            entries = int(row["entries"])
            for _ in range(entries):
                uid = uid_counter
                uid_counter += 1

                # chamada correta com 4 argumentos
                env.process(self.driver(env, uid, lot, stats))

                stats.arrivals += 1

            # ------------------------------------------------------
            # SAÍDAS (exits)
            # ------------------------------------------------------
            exits = int(row["exits"])
            for _ in range(exits):
                if lot.res.count > 0:
                    lot.res.release(lot.res.users[0])

            current_index += 1



    # ----------------------------------------------------------
    # DRIVER
    # ----------------------------------------------------------
    def driver(self, env, uid, lot, stats):
        arrival = env.now

        lot.occ_update(env)
        with lot.res.request() as req:
            yield req

            # realistic stay duration for parking P023
            mean_minutes = 120   # 2 hours average stay
            sigma = 0.6

            mu = np.log(mean_minutes) - (sigma**2)/2
            dur = float(np.random.lognormal(mu, sigma))

            yield env.timeout(dur)

            lot.occ_update(env)
            stats.served += 1
            stats.rev_park += lot.price
            stats.revenue = stats.rev_park
            stats.soj.append(env.now - arrival)

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

        # Process arrivals reconstructed from occupancy forecast
        env.process(self.arrivals_from_forecast(env, lot, stats, df_arrivals))

        # Sampling timeline every 5 minutes: occupancy, price, EV utilization
        env.process(self.sampler(env, lot, stats))

        # Total simulation time: last absolute minute + buffer
        T = int(df_arrivals["abs_min"].max() + 60)

        # Run the simulation
        env.run(until=T)

        # Return results using Stats.to_dict (expects list of lots)
        return stats.to_dict([lot], T), df_arrivals

