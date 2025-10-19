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

    # New attributes for extended behavior
    type: str = "generic"
    amenities_score: float = 0.5
    weekend_multiplier: float = 1.0
    price_min: float = 0.8
    price_cap: float = 4.0

    def build(self, env):
        self.res = simpy.Resource(env, capacity=self.capacity)
        self.ev_res = simpy.Resource(env, capacity=self.ev_spots) if self.ev_spots > 0 else None
        self.charge_time = 0.0
        self.last = env.now
        self.occ_time = 0.0

    def occ_update(self, env):
        dt = env.now - self.last
        if dt > 0:
            self.occ_time += dt * self.res.count
            if self.ev_res:
                self.charge_time += dt * self.ev_res.count
        self.last = env.now

class Stats:
    def __init__(self):
        self.revenue = 0.0
        self.rev_park = 0.0
        self.rev_energy = 0.0
        self.rejected = 0
        self.redirected = 0
        self.redirect_success = 0
        self.redirect_fail = 0
        self.rej_no_spot = 0
        self.rej_no_ev = 0
        self.wait = []
        self.soj = []
        self.arrivals = 0
        self.served = 0
        self.energy_delivered = 0.0
        self.ev_served = 0
        self.ev_want = 0
        self.ev_got = 0
        self.price_sum = {}
        self.price_n = {}
        self.ts = []
        self.profit = 0.0
        self.energy_cost = 0.0
        self.maintenance_cost = 0.0


    def to_dict(self, lots, T):
        occ = {L.name: L.occ_time / (T * L.capacity) for L in lots}
        avg_price = {k: (self.price_sum[k] / max(1, self.price_n[k])) for k in self.price_sum}
        rej_rate = self.rejected / max(1, self.arrivals)
        occ_global = float(np.mean(list(occ.values()))) if occ else 0.0

        wait_p50 = float(np.percentile(self.wait, 50)) if self.wait else 0.0
        wait_p95 = float(np.percentile(self.wait, 95)) if self.wait else 0.0
        redir_rate = self.redirect_success / max(1, self.redirected)

        return {
            "revenue_total": self.revenue,
            "revenue_parking": self.rev_park,
            "revenue_energy": self.rev_energy,
            "revenue_per_car": self.revenue / max(1, self.served),
            "rejected": self.rejected,
            "rej_no_spot": self.rej_no_spot,
            "rej_no_ev": self.rej_no_ev,
            "rejection_rate": rej_rate,
            "wait_avg": float(np.mean(self.wait)) if self.wait else 0.0,
            "wait_p50": wait_p50,
            "wait_p95": wait_p95,
            "sojourn_avg": float(np.mean(self.soj)) if self.soj else 0.0,
            "served": self.served,
            "arrivals": self.arrivals,
            "occ_lots": occ,
            "occ_global": occ_global,
            "avg_price": avg_price,
            "energy_total": self.energy_delivered,
            "ev_served": self.ev_served,
            "ev_want": self.ev_want,
            "ev_got": self.ev_got,
            "timeline": self.ts,  # Time-series data for plots
            "redirected": self.redirected,
            "redirect_success": self.redirect_success,
            "redirect_fail": self.redirect_fail,
            "redirect_success_rate": redir_rate,

    }


# --- Dynamic energy price by time of day ---
# --- Smooth dynamic energy price (€/kWh) ---
def energy_price(t_min, weekend=False,
                 p_base=0.22,       # baseline
                 amp1=0.06,         # main daily amplitude
                 amp2=0.02,         # secondary harmonic
                 phase1=14.0,       # peak ~14:00
                 phase2=20.0,       # secondary ~20:00
                 p_min=0.14, p_max=0.36,
                 ar_rho=0.9):
    """Continuous price with daily harmonics + AR(1) noise. t_min is minutes from 0."""
    h = (t_min / 60.0) % 24.0
    # deterministic signal: 2 harmonics
    import math
    s1 = amp1 * math.sin(2*math.pi*(h - phase1)/24.0)
    s2 = amp2 * math.sin(4*math.pi*(h - phase2)/24.0)
    # weekend discount
    wdisc = -0.02 if weekend else 0.0
    # AR(1) noise cached on function attribute
    z_prev = getattr(energy_price, "_z", 0.0)
    eps = 0.005 * (2*random.random()-1)  # tiny shock
    z = ar_rho*z_prev + eps
    energy_price._z = z
    p = p_base + s1 + s2 + wdisc + z
    return max(p_min, min(p_max, p))

# --- Time-varying arrival rate (arrivals per minute) ---
# --- Determine season by day of year ---
def season_of_day(t_min):
    """Return season based on day of year (assuming 365 days)."""
    day = int(t_min // 1440)
    if   0 <= day < 79:   return "winter"
    elif 79 <= day < 171: return "spring"
    elif 171 <= day < 263:return "summer"
    elif 263 <= day < 354:return "autumn"
    else:                 return "winter"

# --- Time-varying arrival rate with seasonal and event modifiers ---
def lambda_of_t(t_min, cfg):
    """Piecewise-smooth arrival rate, modulated by season and events."""
    h = (t_min / 60.0) % 24.0
    # base shape (0–24h)
    base = (0.2 if h < 6 else
            1.2 + 0.6*(h-6) if h < 9 else
            3.0 if h < 12 else
            2.2 if h < 16 else
            2.6 if h < 20 else
            0.8)

    # --- Seasonal modifier ---
    season = season_of_day(t_min)
    mult = cfg.get("season_multiplier", {}).get(season, 1.0)

    # --- Event effects ---
    for e in cfg.get("event_calendar", []):
        day_start = e["day"] * 1440
        if day_start <= t_min < day_start + 1440:
            hour = (t_min % 1440) / 60.0
            if e["start_hour"] <= hour < e["end_hour"]:
                return base * mult * e.get("demand_uplift", 1.0)

    return base * mult

def arrivals_nhpp(env, lots, policy, stats, sample_duration, max_wait, cfg):
    """Ogata thinning for non-homogeneous Poisson process."""
    i = 0
    t = env.now
    stats.arrivals = 0 
    lam_max = cfg.get("lambda_max", 3.5)  # >= max lambda_of_t over day
    T = cfg.get("T", cfg.get("T_days", 1) * 1440)
    while True:
        # candidate interarrival ~ Exp(lam_max)
        ia = random.expovariate(lam_max)
        t += ia
        if t > T:
            break

        # accept with prob lambda(t)/lam_max
        if random.random() < (lambda_of_t(t, cfg) / lam_max):

            yield env.timeout(max(0.0, t - env.now))
            stats.arrivals += 1
            env.process(driver(env, i, lots, policy, stats, sample_duration, max_wait, cfg))
            i += 1
        else:
            # rejected candidate, skip to next loop iteration
            pass

def find_closest_available(lots, current_lot, max_distance=2.0):
    """Return closest alternative lot with available spots, or None if none within threshold."""
    available = [L for L in lots if L.res.count < L.capacity and L.name != current_lot.name]
    if not available:
        return None
    closest = min(available, key=lambda L: L.distance)
    if abs(closest.distance - current_lot.distance) <= max_distance:
        return closest
    return None

def driver(env, i, lots, policy, stats, sample_duration, max_wait, cfg):
    """One driver process: chooses lot, may redirect, may charge if EV."""
    arrival = env.now
    is_ev = random.random() < float(cfg.get("ev_share", 0))

    # --- Driver chooses an initial lot (may be full) ---
    lot = policy.choose(lots, now=env.now)

    # --- Case 1: No lot selected at all ---
    if lot is None:
        stats.rejected += 1
        stats.rej_no_spot += 1
        alt = find_closest_available(lots, None, max_distance=3.0)
        if alt:
            stats.redirected += 1
            yield env.timeout(abs(alt.distance) * 2.0)  # travel time
            stats.redirect_success += 1
            lot = alt
        else:
            stats.redirect_fail += 1
            return

    # --- Try to park in chosen lot ---
    lot.occ_update(env)
    with lot.res.request() as req:
        start = env.now
        mw = policy.max_wait() if hasattr(policy, "max_wait") else max_wait
        res = yield req | env.timeout(mw)
        waited = env.now - start

        # --- Case 2: Spot unavailable, attempt redirect ---
        if req not in res:
            alt = find_closest_available(lots, lot, max_distance=3.0)
            if alt:
                stats.redirected += 1
                delay = abs(alt.distance - lot.distance) * 2.0
                yield env.timeout(delay)
                p_accept = max(0.0, 1.0 - 0.1 * abs(alt.distance - lot.distance))
                if random.random() < p_accept:
                    stats.redirect_success += 1
                    lot = alt
                    lot.occ_update(env)
                    # retry parking once
                    with lot.res.request() as req2:
                        res2 = yield req2 | env.timeout(mw)
                        if req2 not in res2:
                            stats.rejected += 1
                            stats.rej_no_spot += 1
                            stats.redirect_fail += 1
                            return
                else:
                    stats.redirect_fail += 1
                    stats.rejected += 1
                    stats.rej_no_spot += 1
                    return
            else:
                stats.rejected += 1
                stats.rej_no_spot += 1
                return

        # --- Normal parking process ---
        stats.wait.append(waited)
        dur = float(sample_duration())

        if is_ev and getattr(lot, "ev_res", None) is not None:
            soc_init = float(np.clip(np.random.normal(cfg.get("battery_init_mean", 0.4), 0.05), 0.0, 0.99))
            soc_target = float(np.clip(cfg.get("battery_target", 0.8), 0.0, 1.0))
            want_charge = (random.random() < 0.7) or (soc_init < 0.3)

            if want_charge:
                stats.ev_want += 1
                with lot.ev_res.request() as evreq:
                    ev_res = yield evreq | env.timeout(mw)
                    if evreq not in ev_res:
                        # --- Try redirect if no EV spot available ---
                        alt = find_closest_available(lots, lot, max_distance=3.0)
                        if alt and getattr(alt, "ev_res", None):
                            stats.redirected += 1
                            yield env.timeout(abs(alt.distance - lot.distance) * 2.0)
                            stats.redirect_success += 1
                            lot = alt
                            with lot.ev_res.request() as evreq2:
                                ev_res2 = yield evreq2 | env.timeout(mw)
                                if evreq2 not in ev_res2:
                                    stats.rejected += 1
                                    stats.rej_no_ev += 1
                                    return
                        else:
                            stats.rejected += 1
                            stats.rej_no_ev += 1
                            return

                    # --- Charging phase ---
                    battery_capacity = cfg.get("battery_capacity_kWh", 50.0)
                    energy_needed = (soc_target - soc_init) * battery_capacity  # kWh
                    power_rate = float(cfg.get("power_rate_kW", 7.0))           # kW
                    charge_time = (energy_needed / max(power_rate, 1e-9)) * 60.0  # minutes

                    # define actual_time BEFORE using it
                    actual_time = min(dur, charge_time)

                    # Bill energy hour-by-hour with variable price
                    remaining_time = actual_time
                    energy_delivered = 0.0

                    while remaining_time > 0:
                        step = min(60, remaining_time)            # minutes
                        price_now = energy_price(env.now)         # €/kWh
                        energy_step = (step / 60.0) * power_rate  # kWh
                        stats.rev_energy += energy_step * price_now
                        energy_delivered += energy_step
                        yield env.timeout(step)
                        remaining_time -= step

                    lot.occ_update(env)
                    stats.energy_delivered += energy_delivered
                    stats.ev_served += 1
                    stats.ev_got += 1
                    stats.rev_park += lot.price
                    stats.revenue = stats.rev_park + stats.rev_energy

                    rem = dur - actual_time
                    if rem > 0:
                        yield env.timeout(rem)
                        lot.occ_update(env)
                    stats.energy_delivered += energy_delivered
                    stats.ev_served += 1
                    stats.ev_got += 1
                    stats.rev_park += lot.price
                    stats.revenue = stats.rev_park + stats.rev_energy


                    #print(f"EV #{i} charged {energy:.2f} kWh at {price_now:.2f} €/kWh (hour={env.now/60:.1f})")

                    
            else:
                yield env.timeout(dur)
                lot.occ_update(env)
                stats.rev_park += lot.price
                stats.revenue = stats.rev_park + stats.rev_energy
        else:
            yield env.timeout(dur)
            lot.occ_update(env)
            stats.rev_park += lot.price
            stats.revenue = stats.rev_park + stats.rev_energy

        stats.soj.append(env.now - arrival)
        stats.served += 1
        


def sampler(env, lots, stats, step=5):
    """Sample system state every 'step' minutes for time-series analysis."""
    while True:
        row = {"t": env.now}
        for L in lots:
            occ = L.res.count / max(1, L.capacity)
            row[f"occ_{L.name}"] = occ
            row[f"price_{L.name}"] = L.price
            ev_util = (L.ev_res.count / L.ev_res.capacity) if getattr(L, "ev_res", None) else 0.0
            row[f"evutil_{L.name}"] = ev_util
        stats.ts.append(row)
        yield env.timeout(step)


def run_once(cfg, policy, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    stats = Stats()

    # --- Load global elasticities and seasonal factors ---
    elastic = cfg.get("elasticity", {})
    season_mult = cfg.get("season_multiplier", {})
    season = cfg.get("season", "summer")

    print(f"[INFO] Season: {season}  |  Seasonal multiplier: {season_mult.get(season, 1.0)}")
    print(f"[INFO] Elasticities: {elastic}")

    # --- Initialize parking lots ---
    
    lots = [Lot(**d) for d in cfg["lots"]]
    for L in lots:
        L.build(env)
        stats.price_sum[L.name] = 0.0
        stats.price_n[L.name] = 0

    if hasattr(policy, "schedule"):
        env.process(policy.schedule(env, lots, stats))

    m, s = cfg["duration_lognorm"]["mean"], cfg["duration_lognorm"]["sigma"]
    sample_duration = lambda: np.random.lognormal(m, s)

    env.process(arrivals_nhpp(env, lots, policy, stats, sample_duration, cfg["max_wait"], cfg))

    env.process(sampler(env, lots, stats, step=5))
    T = cfg.get("T", cfg.get("T_days", 1) * 1440)
    env.run(until=T)

    
    return stats.to_dict(lots, T)

