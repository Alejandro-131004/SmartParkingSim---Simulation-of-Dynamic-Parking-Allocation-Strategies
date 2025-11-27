import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass


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

class TwinStats:
    def __init__(self):
        self.revenue = 0.0
        self.rev_park = 0.0
        self.served = 0
        self.arrivals = 0
        self.soj = []
        self.ts = []

    def to_dict(self, lot, T):
        occ = lot.occ_time / (T * lot.capacity)
        soj_avg = float(np.mean(self.soj)) if self.soj else 0.0

        return {
            "served": self.served,
            "arrivals": self.arrivals,
            "revenue": self.revenue,
            "sojourn_avg": soj_avg,
            "occ_lot": occ,
            "timeline": self.ts
        }


# ============================================================
# OFFLINE DIGITAL TWIN SIMULATOR
# ============================================================

class OfflineDigitalTwinSimulator:

    def __init__(self, cfg):
        self.cfg = cfg

    # ----------------------------------------------------------
    # FORECAST OCC → ARRIVALS
    # ----------------------------------------------------------
    def occupancy_to_arrivals(self, df_pred: pd.DataFrame) -> pd.DataFrame:
        df_real = pd.read_csv(self.cfg["dataset_path"])
        df_real.columns = [c.strip().lower() for c in df_real.columns]

        df_real["date"] = pd.to_datetime(df_real["date"]).dt.date
        df_pred["date"] = pd.to_datetime(df_pred["date"]).dt.date

        df_real["hour"] = df_real["hour"].astype(int)
        df_pred["hour"] = df_pred["hour"].astype(int)

        # Set correct parking lot
        parking_id = self.cfg.get("parking_lot_id", "P023")
        df_real = df_real[df_real["parking_lot"] == parking_id].copy()
        df_pred["parking_lot"] = parking_id  # <--- CRUCIAL

        # Keep only forecast rows matching the real dataset in date
        df_pred = df_pred[df_pred["date"].isin(df_real["date"].unique())]

        # Merge using full key (date, hour, and parking lot)
        if "exits" in df_real.columns:
            print("[DEBUG] Using real exits column.")
            df = df_pred.merge(
                df_real[["date", "hour", "parking_lot", "exits"]],
                on=["date", "hour", "parking_lot"],
                how="left"
            )
        else:
            print("[WARNING] No 'exits' column found. Setting exits = 0.")
            df = df_pred.copy()
            df["exits"] = 0

        # Ensure exits exists
        df["exits"] = df.get("exits", 0)
        if isinstance(df["exits"], int):
            df["exits"] = 0
        df["exits"] = df["exits"].fillna(0)

        df["delta_occupancy"] = df["forecast_occ"].diff().fillna(0)
        # Compute estimated arrivals
        df["arrivals"] = (df["delta_occupancy"] + df["exits"]).clip(lower=0).astype(int)

        # Define the absolute time in minutes since the start of the test window
        df["abs_min"] = ((pd.to_datetime(df["date"]) - pd.to_datetime(df["date"]).min()).dt.days * 24 + df["hour"]) * 60

        return df[["date", "hour", "parking_lot", "forecast_occ", "exits", "arrivals", "abs_min"]]




    # ----------------------------------------------------------
    # PROCESS ARRIVALS
    # ----------------------------------------------------------
    def arrivals_from_forecast(self, env, lot, stats, df_arr):
        df_arr = df_arr.sort_values(["date", "hour"])
        prev_abs = None

        for _, row in df_arr.iterrows():
            abs_min = int(row["abs_min"])
            arr = int(row["arrivals"])

            if prev_abs is not None:
                gap = abs_min - prev_abs - 60
                if gap > 0:
                    yield env.timeout(gap)

            prev_abs = abs_min

            if arr > 0:
                interval = 60.0 / arr
                for i in range(arr):
                    stats.arrivals += 1
                    env.process(self.driver(env, f"a_{abs_min}_{i}", lot, stats))
                    yield env.timeout(interval)

    # ----------------------------------------------------------
    # DRIVER
    # ----------------------------------------------------------
    def driver(self, env, uid, lot, stats):
        arrival = env.now

        lot.occ_update(env)
        with lot.res.request() as req:
            yield req

            dur = float(np.random.lognormal(2.0, 0.6))
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
            stats.ts.append({"t": env.now, "occ": lot.res.count / lot.capacity})
            yield env.timeout(5)

    # ----------------------------------------------------------
    # RUN SIMULATION
    # ----------------------------------------------------------
    def run_simulation(self, df_arrivals, policy=None):

        env = simpy.Environment()
        stats = TwinStats()

        lot_cfg = next(l for l in self.cfg["lots"] if l["name"] == "P023")
        # Keep only valid TwinLot arguments
        valid_lot_keys = ["name", "capacity", "price", "distance"]
        lot_cfg = {k: v for k, v in lot_cfg.items() if k in valid_lot_keys}
        lot = TwinLot(**lot_cfg)
        lot.build(env)

        env.process(self.arrivals_from_forecast(env, lot, stats, df_arrivals))
        env.process(self.sampler(env, lot, stats))

        T = int(df_arrivals["abs_min"].max() + 60)
        env.run(until=T)

        return stats.to_dict(lot, T), df_arrivals
