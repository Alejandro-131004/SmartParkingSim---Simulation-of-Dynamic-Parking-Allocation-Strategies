import math, random

# ---------- Base helpers ----------
def load(L):
    """Instantaneous utilization ratio of lot."""
    return L.res.count / max(1, L.capacity)

# ---------- FCFS ----------
class FCFS:
    def choose(self, lots, now):
        # choose by price, then distance — IGNORE capacity here
        return sorted(lots, key=lambda L: (L.price, L.distance))[0] if lots else None

    def max_wait(self): return 10
    def pay(self, lot, payment): return lot.price

# ---------- BalancedCost ----------
class BalancedCost:
    def __init__(self, alpha=0.1, beta=2.5, max_wait_min=10):
        self.alpha, self.beta, self._mw = alpha, beta, max_wait_min

    def max_wait(self): return self._mw

    def choose(self, lots, now):
        if not lots: return None
        def load(L): return L.res.count / max(1, L.capacity)
        def cost(L): return L.price + self.alpha*L.distance + self.beta*load(L)
        return min(lots, key=cost)

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

    def max_wait(self): return self._mw

    def _occ(self, L): return L.res.count / max(1, L.capacity)

    def choose(self, lots, now):
        if not lots: return None
        def occ(L): return L.res.count / max(1, L.capacity)
        def cost(L): return L.price + self.alpha*L.distance + self.beta*occ(L)
        return min(lots, key=cost)

    def pay(self, lot, payment): return lot.price

    def schedule(self, env, lots, stats):
        while True:
            for L in lots:
                occ = self._occ(L)
                new_p = L.price + self.k * (occ - self.target)
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
