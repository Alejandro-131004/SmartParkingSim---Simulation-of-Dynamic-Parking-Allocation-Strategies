import math, random

# ---------- Base helpers ----------
def load(L):
    """Instantaneous utilization ratio of a lot."""
    return L.res.count / max(1, L.capacity)

# -------------------------------
# Simple internal occupancy forecaster
# -------------------------------

def predict_future_occupancy(lot, horizon=60, window=6):
    """Predict future occupancy using a moving average."""
    if not hasattr(lot, "_occ_history"):
        lot._occ_history = []
    occ_now = lot.res.count / max(1, lot.capacity)
    lot._occ_history.append(occ_now)
    if len(lot._occ_history) > window:
        lot._occ_history = lot._occ_history[-window:]
    return sum(lot._occ_history) / len(lot._occ_history)

# ---------- FCFS ----------
class FCFS:
    """First-Come, First-Served: chooses cheapest, closest lot."""
    def choose(self, lots, now):
        return sorted(lots, key=lambda L: (L.price, L.distance))[0] if lots else None
    def max_wait(self): return 10
    def pay(self, lot, payment): return lot.price

# ---------- BalancedCost ----------
class BalancedCost:
    """Chooses lot minimizing combined cost of price, distance, and occupancy."""
    def __init__(self, alpha=0.1, beta=2.5, max_wait_min=10):
        self.alpha, self.beta, self._mw = alpha, beta, max_wait_min
    def max_wait(self): return self._mw
    def choose(self, lots, now):
        if not lots: return None
        return min(lots, key=lambda L: L.price + self.alpha * L.distance + self.beta * load(L))
    def pay(self, lot, payment): return lot.price

# ---------- DynamicPricingBalanced ----------
class DynamicPricingBalanced:
    def __init__(self, alpha=0.1, beta=2.5, target=0.8, k=1,
                 p_min=1.5, p_max=4.0, interval=2, max_wait_min=10):
        self.alpha, self.beta = alpha, beta
        self.target, self.k = target, k
        self.p_min, self.p_max = p_min, p_max
        self.interval = interval
        self._mw = max_wait_min
        self.elasticities = {}
        self.event_calendar = []

    def set_elasticities(self, elasticities):
        self.elasticities = elasticities or {}

    def set_event_calendar(self, event_calendar):
        self.event_calendar = event_calendar or []

    def max_wait(self): return self._mw

    def _occ(self, L):
        return L.res.count / max(1, L.capacity)

    def _event_modifier(self, now, lot_name):
        for e in self.event_calendar:
            day_start = e["day"] * 1440
            if day_start <= now < day_start + 1440 and e["lot"] == lot_name:
                hour = (now % 1440) / 60.0
                if e["start_hour"] <= hour < e["end_hour"]:
                    return e.get("demand_uplift", 1.0), e.get("price_multiplier", 1.0)
        return 1.0, 1.0

    def choose(self, lots, now):
        if not lots: return None
        e = self.elasticities or {"price": 1.0, "distance": 1.0, "occupancy": 1.0}
        def cost(L):
            uplift, _ = self._event_modifier(now, L.name)
            return (e["price"] * L.price + e["distance"] * L.distance + e["occupancy"] * self._occ(L)) / uplift
        return min(lots, key=cost)

    def pay(self, lot, payment): return lot.price

    def schedule(self, env, lots, stats):
        while True:
            for L in lots:
                forecast_occ = predict_future_occupancy(L, horizon=60)
                new_price = L.price + self.k * (forecast_occ - self.target)
                
                # Event multiplier
                for e in self.event_calendar:
                    day_start = e["day"] * 1440
                    if day_start <= env.now < day_start + 1440 and e["lot"] == L.name:
                        hour = (env.now % 1440) / 60.0
                        if e["start_hour"] <= hour < e["end_hour"]:
                            new_price *= e.get("price_multiplier", 1.0)
                            
                L.price = max(self.p_min, min(self.p_max, new_price))
                stats.price_sum[L.name] += L.price
                stats.price_n[L.name] += 1
            yield env.timeout(self.interval)

# ---------- StaticPricing (NEW) ----------
class StaticPricing(DynamicPricingBalanced):
    """
    Static Pricing Policy.
    - Uses the same driver choice logic (elasticities) as Dynamic.
    - BUT prices NEVER change (schedule is disabled).
    """
    def schedule(self, env, lots, stats):
        # Do nothing. Wait forever. Prices stay at initial config.
        yield env.timeout(float('inf'))

# ---------- Factory ----------
def make_policy(name):
    if name == "fcfs": return FCFS()
    if name == "balanced": return BalancedCost()
    if name == "dynpricing": return DynamicPricingBalanced()
    if name == "static": return StaticPricing()  # <--- FIXED: Added handling for 'static'
    raise ValueError(f"Unknown policy: {name}")

# ---------- Probabilistic selector ----------
def choose_probabilistic(lots, alpha=0.1, beta=2.0):
    avail = [L for L in lots if L.res.count < L.capacity]
    if not avail: return None
    utils = [-(L.price + alpha * L.distance + beta * load(L)) for L in avail]
    m = max(utils)
    ps = [math.exp(u - m) for u in utils]
    s = sum(ps)
    r = random.random() * s
    acc = 0.0
    for L, p in zip(avail, ps):
        acc += p
        if r <= acc: return L
    return avail[0]