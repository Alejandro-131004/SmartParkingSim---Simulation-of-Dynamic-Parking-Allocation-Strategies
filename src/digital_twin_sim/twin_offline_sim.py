import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ============================================================
# Parking lot object (offline twin)
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
# Stats container
# ============================================================

class TwinStats:
    def __init__(self):
        self.revenue = 0.0
        self.rev_park = 0.0
        self.rejected = 0
        self.served = 0
        self.arrivals = 0
        self.wait = []
        self.soj = []
        self.ts = []

    def to_dict(self, lot, T):
        occ = lot.occ_time / (T * lot.capacity)
        wait_avg = float(np.mean(self.wait)) if self.wait else 0.0
        soj_avg = float(np.mean(self.soj)) if self.soj else 0.0

        return {
            "served": self.served,
            "arrivals": self.arrivals,
            "revenue": self.revenue,
            "wait_avg": wait_avg,
            "sojourn_avg": soj_avg,
            "occ_lot": occ,
            "timeline": self.ts
        }


# ============================================================
# Offline Twin - Convert Forecast → Arrivals
# ============================================================

class OfflineDigitalTwinSimulator:

    def __init__(self, cfg):
        self.cfg = cfg

    # ----------------------------------------------------------
    # Convert forecasted OCCUPANCY → forecasted ARRIVALS
    # ----------------------------------------------------------
    def occupancy_to_arrivals(self, df_pred):
        """
        Convert predicted occupancy (absolute count) into arrivals
        using Δocc + exits.
        """

        df = df_pred.copy()
        df = df.sort_values(["date", "hour"])

        df["delta_occ"] = df["forecast_occ"].diff().fillna(0)

        # Get real exits from the dataset
        df_real = pd.read_csv(self.cfg["dataset_path"])
        df_real.columns = [c.strip().lower() for c in df_real.columns]
        df_real["date"] = pd.to_datetime(df_real["date"])

        df = df.merge(
            df_real[["date", "hour", "exits"]],
            on=["date", "hour"],
            how="left"
        )

        df["exits"] = df["exits"].fillna(0)

        # arrivals ≈ Δocc + exits
        arrivals = df["delta_occ"] + df["exits"]
        arrivals = arrivals.clip(lower=0).round().astype(int)

        df["arrivals"] = arrivals

        # ----------------------------------------------------------
        # REQUIRED FIX → simulation horizon column
        # ----------------------------------------------------------
        df["abs_min"] = (
            (df["date"] - df["date"].min()).dt.days * 1440
            + df["hour"].astype(int) * 60
        )

        return df

    # ----------------------------------------------------------
    # Simulate arrivals as a stochastic process in SimPy
    # ----------------------------------------------------------
    def arrivals_from_forecast(self, env, lot, stats, df_arr):
        df_arr = df_arr.sort_values(["date", "hour"])

        prev_abs = None

        for _, row in df_arr.iterrows():
            abs_min = int(row["abs_min"])
            arr = int(row["arrivals"])

            # time gap
            if prev_abs is not None:
                gap = abs_min - prev_abs - 60
                if gap > 0:
                    yield env.timeout(gap)

            prev_abs = abs_min

            # distribute arrivals
            if arr > 0:
                interval = 60.0 / arr
                for i in range(arr):
                    stats.arrivals += 1
                    env.process(
                        self.driver(env, f"P023_{abs_min}_{i}", lot, stats)
                    )
                    yield env.timeout(interval)

    # ----------------------------------------------------------
    # Driver Process (simplified offline version)
    # ----------------------------------------------------------
    def driver(self, env, uid, lot, stats):
        arrival = env.now

        lot.occ_update(env)
        with lot.res.request() as req:
            yield req

            # duration
            dur = float(np.random.lognormal(2.0, 0.6))
            yield env.timeout(dur)
            lot.occ_update(env)

            stats.served += 1
            stats.rev_park += lot.price
            stats.revenue = stats.rev_park
            stats.soj.append(env.now - arrival)

    # ----------------------------------------------------------
    # Sampling time-series
    # ----------------------------------------------------------
    def sampler(self, env, lot, stats):
        while True:
            stats.ts.append({
                "t": env.now,
                "occ": lot.res.count / lot.capacity,
                "price": lot.price
            })
            yield env.timeout(5)

    # ----------------------------------------------------------
    # Main function to run SimPy simulation from forecasted arrivals
    # ----------------------------------------------------------
    def run_simulation(self, df_arrivals, policy=None):
        """
        df_arrivals MUST contain:
        - date
        - hour
        - arrivals
        - abs_min
        """

        env = simpy.Environment()
        stats = TwinStats()

        # Only one lot: P023
        lot_cfg = next(l for l in self.cfg["lots"] if l["name"] == "P023")
        lot = TwinLot(**lot_cfg)
        lot.build(env)

        # Start processes
        env.process(self.arrivals_from_forecast(env, lot, stats, df_arrivals))
        env.process(self.sampler(env, lot, stats))

        # Compute horizon
        T = int(df_arrivals["abs_min"].max() + 60)

        env.run(until=T)

        return stats.to_dict(lot, T), df_arrivals
