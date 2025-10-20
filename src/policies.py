import math, random

# ---------- Base helpers ----------
def load(L):
    """Instantaneous utilization ratio of a lot."""
    return L.res.count / max(1, L.capacity)

# ---------- FCFS ----------
class FCFS:
    """First-Come, First-Served: chooses cheapest, closest lot."""
    def choose(self, lots, now):
        return sorted(lots, key=lambda L: (L.price, L.distance))[0] if lots else None

    def max_wait(self): 
        return 10

    def pay(self, lot, payment): 
        return lot.price

# ---------- BalancedCost ----------
class BalancedCost:
    """Chooses lot minimizing combined cost of price, distance, and occupancy."""
    def __init__(self, alpha=0.1, beta=2.5, max_wait_min=10):
        self.alpha, self.beta, self._mw = alpha, beta, max_wait_min

    def max_wait(self): 
        return self._mw

    def choose(self, lots, now):
        if not lots: 
            return None

        def load(L): 
            return L.res.count / max(1, L.capacity)

        def cost(L): 
            return L.price + self.alpha * L.distance + self.beta * load(L)

        return min(lots, key=cost)

    def pay(self, lot, payment): 
        return lot.price

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
        """Attach elasticity parameters from config."""
        self.elasticities = elasticities or {}

    def set_event_calendar(self, event_calendar):
        """Attach event calendar from config."""
        self.event_calendar = event_calendar or []

    def max_wait(self):
        return self._mw

    def _occ(self, L):
        return L.res.count / max(1, L.capacity)

    # --- NEW: detect active event and return uplift factor ---
    def _event_modifier(self, now, lot_name):
        for e in self.event_calendar:
            day_start = e["day"] * 1440
            if day_start <= now < day_start + 1440 and e["lot"] == lot_name:
                hour = (now % 1440) / 60.0
                if e["start_hour"] <= hour < e["end_hour"]:
                    uplift = e.get("demand_uplift", 1.0)
                    price_mult = e.get("price_multiplier", 1.0)
                    # Debug print for validation
                    print(f"[EVENT ACTIVE] Day {e['day']} | Lot {lot_name} | "
                          f"{e['start_hour']}-{e['end_hour']}h | "
                          f"Demand x{uplift} | Price x{price_mult}")
                    return uplift, price_mult
        return 1.0, 1.0

    def choose(self, lots, now):
        """Driver chooses lot based on weighted cost (elasticities + events)."""
        if not lots:
            return None

        e = self.elasticities or {"price": 1.0, "distance": 1.0, "occupancy": 1.0}

        def cost(L):
            occ = self._occ(L)
            uplift, _ = self._event_modifier(now, L.name)
            return (e["price"] * L.price +
                    e["distance"] * L.distance +
                    e["occupancy"] * occ) / uplift

        return min(lots, key=cost)

    def pay(self, lot, payment):
        return lot.price

    def schedule(self, env, lots, stats):
        """Periodically update lot prices based on occupancy and events."""
        while True:
            for L in lots:
                occ = self._occ(L)
                new_p = L.price + self.k * (occ - self.target)

                # --- Apply event-based price multiplier dynamically ---
                for e in self.event_calendar:
                    day_start = e["day"] * 1440
                    if day_start <= env.now < day_start + 1440 and e["lot"] == L.name:
                        hour = (env.now % 1440) / 60.0
                        if e["start_hour"] <= hour < e["end_hour"]:
                            mult = e.get("price_multiplier", 1.0)
                            new_p *= mult
                            print(f"[PRICE INCREASE] {L.name} → {round(new_p, 2)} €/h "
                                  f"(Event {e['day']} {e['start_hour']}-{e['end_hour']}h)")

                # Cap and store updated price
                L.price = max(self.p_min, min(self.p_max, new_p))
                stats.price_sum[L.name] += L.price
                stats.price_n[L.name] += 1

            yield env.timeout(self.interval)

# ---------- Probabilistic selector ----------
def choose_probabilistic(lots, alpha=0.1, beta=2.0):
    """Softmax-based probabilistic selection between available lots."""
    avail = [L for L in lots if L.res.count < L.capacity]
    if not avail:
        return None
    utils = [-(L.price + alpha * L.distance + beta * load(L)) for L in avail]
    m = max(utils)
    ps = [math.exp(u - m) for u in utils]
    s = sum(ps)
    r = random.random() * s
    acc = 0.0
    for L, p in zip(avail, ps):
        acc += p
        if r <= acc:
            return L
    return avail[0]

# ---------- Factory ----------
def make_policy(name):
    if name == "fcfs": return FCFS()
    if name == "balanced": return BalancedCost()
    if name == "dynpricing": return DynamicPricingBalanced()
    raise ValueError(f"Unknown policy: {name}")
