import math, random
import numpy as np

# ---------- Base helpers ----------
def load(L):
    """Instantaneous utilization ratio of a lot."""
    return L.res.count / max(1, L.capacity)

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
    def __init__(self, alpha=0.1, beta=2.5, target=0.8, k=1.0,
                 p_min=0.0, p_max=5.0, interval=15, max_wait_min=10):
        self.alpha, self.beta = alpha, beta
        self.target, self.k = target, k
        self.p_min, self.p_max = p_min, p_max
        self.interval = interval
        self._mw = max_wait_min
        self.elasticities = {}
        self.event_calendar = []
        
        self.last_price = {}
        self.last_revenue = {}

    def set_elasticities(self, elasticities):
        self.elasticities = elasticities or {}

    def set_event_calendar(self, event_calendar):
        self.event_calendar = event_calendar or []

    def max_wait(self): return self._mw

    def _occ(self, L):
        return L.res.count / max(1, L.capacity)

    def choose(self, lots, now):
        if not lots: return None
        e = self.elasticities or {"price": 1.0, "distance": 1.0, "occupancy": 1.0}
        
        # Simple event check for uplift (simplified)
        uplift = 1.0
        for ev in self.event_calendar:
            day_start = ev["day"] * 1440
            if day_start <= now < day_start + 1440:
                hour = (now % 1440) / 60.0
                if ev["start_hour"] <= hour < ev["end_hour"]:
                    uplift = ev.get("demand_uplift", 1.0)

        def cost(L):
            return (e["price"] * L.price + e["distance"] * L.distance + e["occupancy"] * self._occ(L)) / uplift
        return min(lots, key=cost)

    def pay(self, lot, payment): return lot.price

    def schedule(self, env, lots, stats):
        # Initialize tracking for existing lots
        for L in lots:
            self.last_price[L.name] = L.price
            self.last_revenue[L.name] = (L.res.count / max(1, L.capacity)) * L.capacity * L.price

        while True:
            for L in lots:
                # ---------------------------------------------------------
                # 0. Measure Current State
                # ---------------------------------------------------------
                current_occ_count = L.res.count
                current_price = L.price
                current_revenue = current_occ_count * current_price
                
                prev_price = self.last_price.get(L.name, current_price)
                prev_revenue = self.last_revenue.get(L.name, current_revenue)
                
                # 1. Forecast Occupancy
                forecast_occ = predict_future_occupancy(L, horizon=60)
                
                # ---------------------------------------------------------
                # 2. REVENUE GUARDRAIL CHECK 
                # ---------------------------------------------------------
                # Logic: If we raised price, but revenue dropped, revert immediately.
                guard_triggered = False
                revenue_drop_threshold = 0.95 
                
                if current_price > prev_price:
                    if current_revenue < prev_revenue * revenue_drop_threshold:
                        # BAD outcome: Price Up -> Revenue Down. Revert!
                        new_price_target = prev_price * 0.9 
                        guard_triggered = True
                
                # ---------------------------------------------------------
                # 3. Integral Control with Low Demand Guardrail
                # ---------------------------------------------------------
                if not guard_triggered:
                    # Guardrail 1: Low Demand (< 30%) -> Block price drops
                    if forecast_occ < 0.30:
                        if current_price < (self.p_max * 0.5):
                            price_change = 0.05
                        else:
                            price_change = 0.0
                    else:
                        # Normal PID Control
                        error = forecast_occ - self.target
                        price_change = error * self.k
                        price_change = max(-0.5, min(0.5, price_change))

                    new_price_target = current_price + price_change

                # 4. Final Updates
                final_price = max(self.p_min, min(self.p_max, new_price_target))
                
                # Store state for next comparison
                self.last_price[L.name] = final_price 
                self.last_revenue[L.name] = current_revenue 
                
                L.price = final_price
                stats.price_sum[L.name] += L.price
                stats.price_n[L.name] += 1
                
            yield env.timeout(self.interval)

# ---------- StaticPricing ----------
class StaticPricing(DynamicPricingBalanced):
    """
    Static Pricing Policy.
    """
    def schedule(self, env, lots, stats):
        yield env.timeout(float('inf'))

# ---------- MCTSPricing ----------
class MCTSPricing:
    """
    Uses Monte Carlo Tree Search to plan pricing.
    """
    def __init__(self, interval=15, p_min=0.0, p_max=5.0):
        self.interval = interval
        self.p_min = p_min
        self.p_max = p_max
        self.agent = None 
        
    def set_forecast(self, df_forecast):
        from mcts_agent import MCTSAgent
        self.agent = MCTSAgent(
            forecast_df=df_forecast, 
            planning_horizon=60, 
            n_simulations=100, 
            p_min=self.p_min,
            p_max=self.p_max
        )

    def choose(self, lots, now):
        return sorted(lots, key=lambda L: (L.price, L.distance))[0] if lots else None

    def pay(self, lot, payment): return lot.price

    def schedule(self, env, lots, stats):
        if self.agent is None:
            print("[WARN] MCTS Policy started without Forecast Data. Pricing will be static.")
            yield env.timeout(float('inf'))
            
        while True:
            for L in lots:
                current_state = {
                    't': int(env.now),
                    'occ': L.res.count,
                    'price': L.price
                }
                
                # Agent returns price DELTA
                price_delta = self.agent.get_action(current_state)
                
                new_price = L.price + price_delta
                final_price = max(self.p_min, min(self.p_max, new_price))
                
                L.price = final_price
                stats.price_sum[L.name] += L.price
                stats.price_n[L.name] += 1
                
            yield env.timeout(self.interval)

# ---------- Factory ----------
def make_policy(name):
    # Handle dict config case
    if isinstance(name, dict):
        p_type = name.get("type", "balanced")
        if p_type == "dynpricing":
            return DynamicPricingBalanced(
                target=float(name.get("target_occupancy", 0.8)),
                k=float(name.get("k", 1.0)),
                p_min=float(name.get("p_min", 0.0))
            )
        if p_type == "mcts":
             return MCTSPricing(
                 p_min=float(name.get("p_min", 0.0))
             )
        name = p_type

    if name == "fcfs": return FCFS()
    if name == "balanced": return BalancedCost()
    if name == "dynpricing": return DynamicPricingBalanced()
    if name == "static": return StaticPricing()
    if name == "mcts": return MCTSPricing()
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