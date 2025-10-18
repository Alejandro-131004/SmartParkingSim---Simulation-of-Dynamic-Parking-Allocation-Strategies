import simpy, numpy as np, random
from dataclasses import dataclass

@dataclass
class Lot:
    name: str
    capacity: int
    price: float
    distance: float
    ev_spots: int = 0
    chargers: int = 0

    def build(self, env):
        # Regular parking spots
        self.res = simpy.Resource(env, capacity=self.capacity)
        # EV charging spots (subset of total capacity)
        self.ev_res = simpy.Resource(env, capacity=self.ev_spots) if self.ev_spots > 0 else None
        # Charger utilization tracking
        self.charge_time = 0.0
        self.last = env.now
        self.occ_time = 0.0

    def occ_update(self, env):
        # Integrate occupied time for regular and EV spots
        dt = env.now - self.last
        if dt > 0:
            self.occ_time += dt * self.res.count
            if self.ev_res:
                self.charge_time += dt * self.ev_res.count
        self.last = env.now

class Stats:
    def __init__(self):
        self.revenue = 0.0
        self.rejected = 0
        self.wait = []
        self.soj = []
        self.arrivals = 0
        self.served = 0
        self.energy_delivered = 0.0   # NEW
        self.ev_served = 0            # NEW
        self.price_sum = {}
        self.price_n = {}

    def to_dict(self, lots, T):
        occ = {L.name: L.occ_time / (T * L.capacity) for L in lots}
        avg_price = {k: (self.price_sum[k] / max(1, self.price_n[k])) for k in self.price_sum}
        rej_rate = self.rejected / max(1, self.arrivals)
        occ_global = float(np.mean(list(occ.values()))) if occ else 0.0
        return {
            "revenue_total": self.revenue,
            "revenue_per_car": self.revenue / max(1, self.served),
            "rejected": self.rejected,
            "rejection_rate": rej_rate,
            "wait_avg": float(np.mean(self.wait)) if self.wait else 0.0,
            "sojourn_avg": float(np.mean(self.soj)) if self.soj else 0.0,
            "served": self.served,
            "arrivals": self.arrivals,
            "occ_lots": occ,
            "occ_global": occ_global,
            "avg_price": avg_price,
            "energy_total": self.energy_delivered,  # NEW
            "ev_served": self.ev_served,            # NEW
        }

def arrivals(env, lam, lots, policy, stats, sample_duration, max_wait, cfg):
    i = 0
    while True:
        ia = random.expovariate(lam)
        yield env.timeout(ia)
        stats.arrivals += 1
        env.process(driver(env, i, lots, policy, stats, sample_duration, max_wait, cfg))

        i += 1

def driver(env, i, lots, policy, stats, sample_duration, max_wait, cfg):
    arrival = env.now
    is_ev = random.random() < cfg.get("ev_share", 0)


    lot = policy.choose(lots, now=env.now)
    if lot is None:
        stats.rejected += 1
        return

    lot.occ_update(env)
    with lot.res.request() as req:
        start = env.now
        mw = policy.max_wait() if hasattr(policy, "max_wait") else max_wait
        res = yield req | env.timeout(mw)
        waited = env.now - start
        if req not in res:
            stats.rejected += 1
            return

        stats.wait.append(waited)
        dur = float(sample_duration())

        if is_ev and getattr(lot, "ev_res", None) is not None:
            if is_ev:
                print(f"EV #{i} assigned to lot {lot.name} at t={env.now}")

            with lot.ev_res.request() as evreq:
                ev_res = yield evreq | env.timeout(mw)
                if evreq not in ev_res:
                    stats.rejected += 1
                    return
                # SoC sampling and clamping
                soc_init = float(np.clip(np.random.normal(cfg.get("battery_init_mean", 0.4), 0.05), 0.0, 0.99))
                soc_target = float(np.clip(cfg.get("battery_target", 0.8), 0.0, 1.0))
                energy_needed = max(0.0, soc_target - soc_init)  # in "SoC units"
                power_rate = float(cfg.get("power_rate", 0.3))   # kWh per minute
                charge_time = energy_needed / max(power_rate, 1e-9)

                actual_time = min(dur, charge_time)
                yield env.timeout(actual_time)
                lot.occ_update(env)

                energy = actual_time * power_rate
                stats.energy_delivered += energy         # NEW
                stats.ev_served += 1                     # NEW
                stats.revenue += lot.price + energy * float(cfg.get("price_per_kWh", 0.25))

                # If parking duration exceeds charging, stay parked for the remaining time
                rem = dur - actual_time
                if rem > 0:
                    yield env.timeout(rem)
                    lot.occ_update(env)
        else:
            # ICE parking
            yield env.timeout(dur)
            lot.occ_update(env)
            stats.revenue += lot.price

        stats.soj.append(env.now - arrival)
        stats.served += 1

def run_once(cfg, policy, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    stats = Stats()

    lots = [Lot(**d) for d in cfg["lots"]]
    for L in lots:
        L.build(env)
        stats.price_sum[L.name] = 0.0
        stats.price_n[L.name] = 0

    if hasattr(policy, "schedule"):
        env.process(policy.schedule(env, lots, stats))

    m, s = cfg["duration_lognorm"]["mean"], cfg["duration_lognorm"]["sigma"]
    sample_duration = lambda: np.random.lognormal(m, s)

    env.process(arrivals(env, cfg["arrival_rate"], lots, policy, stats, sample_duration, cfg["max_wait"], cfg))
    env.run(until=cfg["T"])
    
    return stats.to_dict(lots, cfg["T"])
