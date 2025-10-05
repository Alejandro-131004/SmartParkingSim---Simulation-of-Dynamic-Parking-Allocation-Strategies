class FCFS:
    def choose(self, lots, now):
        # escolhe o parque mais barato; empate resolve pela menor distância
        return sorted(lots, key=lambda L:(L.price, L.distance))[0] if lots else None

def make_policy(name):
    if name == "fcfs": return FCFS()
    raise ValueError(f"Unknown policy: {name}")

def load(L):
    return L.res.count / max(1, L.capacity)

class BalancedCost:
    def __init__(self, alpha=0.1, beta=2.5, max_wait_min=10):
        self.alpha, self.beta, self._mw = alpha, beta, max_wait_min
    def max_wait(self): return self._mw
    def choose(self, lots, now):
        def cost(L):
            return L.price + self.alpha*L.distance + self.beta*load(L)
        return min(lots, key=cost) if lots else None
    def pay(self, lot, payment): return lot.price


    
def make_policy(name):
    if name == "fcfs": return FCFS()
    if name == "balanced": return BalancedCost()
    raise ValueError(f"Unknown policy: {name}")
