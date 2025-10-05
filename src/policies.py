

class FCFS:
    def choose(self, lots, now):
        # Choose the cheapest lot; tie-breaking by shortest distance
        return sorted(lots, key=lambda L: (L.price, L.distance))[0] if lots else None
    def max_wait(self): return 10
    def pay(self, lot, payment): return lot.price


def load(L):
    # Instantaneous utilization: in-service over capacity
    return L.res.count / max(1, L.capacity)


class BalancedCost:
    def __init__(self, alpha=0.1, beta=2.5, max_wait_min=10):
        self.alpha, self.beta, self._mw = alpha, beta, max_wait_min
    def max_wait(self): return self._mw
    def choose(self, lots, now):
        # Cost = price + alpha*distance + beta*load
        def cost(L): return L.price + self.alpha * L.distance + self.beta * load(L)
        return min(lots, key=cost) if lots else None
    def pay(self, lot, payment): return lot.price


class DynamicPricingBalanced:
    def __init__(self, alpha=0.1, beta=2.5, target=0.8, k=1,
                 p_min=1.5, p_max=4.0, interval=2, max_wait_min=10):
        # alpha: distance weight | beta: load weight
        # target: target occupancy | k: proportional controller gain
        self.alpha, self.beta = alpha, beta
        self.target, self.k = target, k
        self.p_min, self.p_max = p_min, p_max
        self.interval = interval          # minutes between price updates
        self._mw = max_wait_min
    def max_wait(self): return self._mw
    def _occ(self, L): return L.res.count / max(1, L.capacity)
    def choose(self, lots, now):
        # Choose by generalized cost with current (dynamic) price
        def cost(L): return L.price + self.alpha * L.distance + self.beta * self._occ(L)
        return min(lots, key=cost) if lots else None
    def pay(self, lot, payment): return lot.price

    def schedule(self, env, lots, stats):
        # Periodically adjust price toward target occupancy and log prices
        while True:
            for L in lots:
                occ = self._occ(L)
                new_p = L.price + self.k * (occ - self.target)
                L.price = max(self.p_min, min(self.p_max, new_p))
                # Log price
                stats.price_sum[L.name] += L.price
                stats.price_n[L.name] += 1
            yield env.timeout(self.interval)



def make_policy(name):
    # Single source of truth
    if name == "fcfs": return FCFS()
    if name == "balanced": return BalancedCost()
    if name == "dynpricing": return DynamicPricingBalanced()
    raise ValueError(f"Unknown policy: {name}")
