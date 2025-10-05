import simpy, numpy as np, random
from dataclasses import dataclass

@dataclass
class Lot:
    name: str; capacity: int; price: float; distance: float
    def build(self, env):
        self.res = simpy.Resource(env, capacity=self.capacity)
        self.last = env.now; self.occ_time = 0.0
    def occ_update(self, env):
        dt = env.now - self.last
        if dt > 0: self.occ_time += dt * self.res.count
        self.last = env.now

class Stats:
    def __init__(self):
        self.revenue=0.0; self.rejected=0; self.wait=[]; self.soj=[]
        self.arrivals=0; self.served=0
    def to_dict(self, lots, T):
        occ = {L.name: L.occ_time/(T*L.capacity) for L in lots}
        rej_rate = self.rejected / max(1, self.arrivals)
        return {
            "revenue": self.revenue,
            "rejected": self.rejected,
            "rejection_rate": rej_rate,
            "wait_avg": float(np.mean(self.wait)) if self.wait else 0.0,
            "sojourn_avg": float(np.mean(self.soj)) if self.soj else 0.0,
            "served": self.served,
            "arrivals": self.arrivals,
            "occ": occ,
        }

def arrivals(env, lam, lots, policy, stats, sample_duration, max_wait):
    i=0
    while True:
        ia = random.expovariate(lam)
        yield env.timeout(ia)
        stats.arrivals += 1
        env.process(driver(env, i, lots, policy, stats, sample_duration, max_wait))
        i += 1

def driver(env, i, lots, policy, stats, sample_duration, max_wait):
    arrival = env.now
    lot = policy.choose(lots, now=env.now)
    if lot is None:
        stats.rejected += 1; return
    lot.occ_update(env)
    with lot.res.request() as req:
        start = env.now
        res = yield req | env.timeout(max_wait)
        waited = env.now - start
        if req not in res:
            stats.rejected += 1; return
        stats.wait.append(waited)
        dur = float(sample_duration())
        yield env.timeout(dur)
        lot.occ_update(env)
        stats.revenue += lot.price
        stats.soj.append(env.now - arrival)
        stats.served += 1

def run_once(cfg, policy, seed):
    random.seed(seed); np.random.seed(seed)
    env = simpy.Environment()
    lots = [Lot(**d) for d in cfg["lots"]]
    for L in lots: L.build(env)
    m, s = cfg["duration_lognorm"]["mean"], cfg["duration_lognorm"]["sigma"]
    sample_duration = lambda: np.random.lognormal(m, s)
    stats = Stats()
    env.process(arrivals(env, cfg["arrival_rate"], lots, policy, stats, sample_duration, cfg["max_wait"]))
    env.run(until=cfg["T"])
    return stats.to_dict(lots, cfg["T"])
