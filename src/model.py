import simpy, numpy as np, random
from dataclasses import dataclass

def _dbg(cfg, *args, **kwargs):
    """Conditional debug print controlled by cfg['debug']."""
    if cfg.get("debug", False):
        print(*args, **kwargs)

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
                # --- Debug counters ---
        self.arrivals_expected_by_lot = {}   # lot -> sum(adj_entries)
        self.arrivals_spawned_by_lot  = {}   # lot -> number of env.process(driver(...)) started
        self.served_by_lot            = {}   # lot -> served count (increment on successful parking end)
        self.rejected_by_lot          = {}   # lot -> rejected count (no spot / no EV)
        self.rejected_reason          = {"no_spot": 0, "no_ev": 0}



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
            "timeline": self.ts,
            "redirected": self.redirected,
            "redirect_success": self.redirect_success,
            "redirect_fail": self.redirect_fail,
            "redirect_success_rate": redir_rate
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
        stats.served_by_lot[lot.name] = stats.served_by_lot.get(lot.name, 0) + 1
        
import pandas as pd
import math, random

def arrivals_from_hourly_dataset(env, lots, policy, stats, cfg, sample_duration):
    """Generate arrivals based on the hourly dataset (one SimPy process per lot)."""
    dataset_path = cfg.get("dataset_path", "dataset_fluxos_hourly_2022.csv")
    df = pd.read_csv(dataset_path)

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"parking_lot", "date", "hour", "entries"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Ensure chronological order per lot
    df = df.sort_values(["parking_lot", "date", "hour"])

    print(f"[INFO] Loaded dataset with {len(df)} hourly records from {dataset_path}")

    # Create a SimPy process per lot
    for lot in lots:
        df_lot = df[df["parking_lot"] == lot.name].copy()
        if df_lot.empty:
            print(f"[WARN] No data found for lot {lot.name}")
            continue
        env.process(arrivals_for_lot(env, lot, df_lot, policy, stats, cfg, sample_duration))


def arrivals_for_lot(env, lot, df_lot, policy, stats, cfg, sample_duration):
    """Simulate arrivals for a single parking lot from its hourly 'entries' series."""
    debug_hours = int(cfg.get("debug_sample_hours", 0))

    elasticity = cfg.get("elasticity", {})
    alpha_price   = float(elasticity.get("price",   1.0))
    alpha_traffic = float(elasticity.get("traffic", 0.0))   # <- usar chave 'traffic' no YAML se quiseres
    alpha_temp    = 0.02  # sensibilidade térmica simples

    base_price = float(lot.price)
    max_wait   = cfg.get("max_wait", 1)

    # Helper para converter (date, hour) numa "hora absoluta" crescente por linha
    # Assim conseguimos detetar hiatos (horas faltantes no CSV)
    df_lot = df_lot.copy()
    df_lot["date"] = pd.to_datetime(df_lot["date"])
    df_lot["abs_hour"] = (df_lot["date"] - df_lot["date"].min()).dt.days * 24 + df_lot["hour"].astype(int)

    last_abs_hour = None
    for _, row in df_lot.iterrows():
        abs_hour = int(row["abs_hour"])
        entries  = int(row["entries"])
        traffic  = float(row.get("traffic_score", 1.0))

        # Temperatura média se existir
        temp_min = row.get("temp_min_c", None)
        temp_max = row.get("temp_max_c", None)
        if pd.notna(temp_min) and pd.notna(temp_max):
            temp_mean = (float(temp_min) + float(temp_max)) / 2.0
        else:
            temp_mean = 22.0  # neutro

        # Se há salto de horas no CSV, avançar o tempo
        if last_abs_hour is None:
            # Primeiro registo do lote: garantir que estamos alinhados ao início
            # (se quiseres alinhar tudo a t=0 neste primeiro registo, não faças timeout aqui)
            pass
        else:
            gap_hours = abs_hour - last_abs_hour - 1
            if gap_hours > 0:
                yield env.timeout(gap_hours * 60)  # minutos

        last_abs_hour = abs_hour

        # Ajuste de intenção por contexto (preço, tráfego, temperatura)
        # Preço: normalizar pela diferença face a base_price (evita divisão por 0)
        if base_price > 0:
            price_factor = max(0.2, 1 - alpha_price * (lot.price - base_price) / base_price)
        else:
            price_factor = 1.0

        # Tráfego (score ~ 0..10): quanto maior o score, maior congestão → ligeira redução
        traffic_factor = max(0.2, 1 - alpha_traffic * (traffic / 10.0))

        # Conforto térmico em torno de 22 ºC
        temp_factor = max(0.4, 1 - alpha_temp * abs(temp_mean - 22.0))

        adj_entries = int(round(entries * price_factor * traffic_factor * temp_factor))

                # track totals
        stats.arrivals_expected_by_lot[lot.name] += adj_entries

        if debug_hours > 0 and (last_abs_hour is None or abs_hour < df_lot["abs_hour"].min() + debug_hours):
            _dbg(cfg, 
                 f"[HOUR] lot={lot.name} abs_hour={abs_hour} "
                 f"entries={entries} adj={adj_entries} "
                 f"factors: price={price_factor:.2f}, traffic={traffic_factor:.2f}, temp={temp_factor:.2f} "
                 f"price_now={lot.price:.2f}")

        if adj_entries <= 0:
            # Mesmo sem chegadas, avançar 1 hora
            yield env.timeout(60)
            continue

        # Espalhar as chegadas uniformemente ao longo da hora
        interval = 60.0 / adj_entries  # minutos por chegada
        for i in range(adj_entries):
            stats.arrivals += 1
            stats.arrivals_spawned_by_lot[lot.name] = stats.arrivals_spawned_by_lot.get(lot.name, 0) + 1
            env.process(driver(
                env,
                f"{lot.name}_{abs_hour}_{i}",
                [lot],
                policy,
                stats,
                sample_duration=sample_duration,
                max_wait=max_wait,
                cfg=cfg
            ))

            yield env.timeout(interval)

        # Ao final dos adj_entries, já somou ~60 minutos (interval * adj_entries ≈ 60)

def _make_sample_duration(cfg):
    """Return a callable duration sampler. Uses YAML if present; otherwise a safe default."""
    # 1) Se existir duration_lognorm no YAML, usa
    dl = cfg.get("duration_lognorm")
    if isinstance(dl, dict) and "mean" in dl and "sigma" in dl:
        m, s = float(dl["mean"]), float(dl["sigma"])
        return lambda: float(np.random.lognormal(m, s))

    # 2) Senão, um default razoável (mediana ~ 60–90 min)
    return lambda: float(np.random.lognormal(2.0, 0.6))


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

def _compute_horizon_minutes(dataset_path: str, cfg: dict) -> int:
    df = pd.read_csv(dataset_path)

    # Basic schema check
    cols = [c.strip().lower() for c in df.columns]
    _dbg(cfg, f"[DATA] Columns: {cols}")

    # Parse times
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    ts = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    start = ts.min()
    end   = ts.max() + pd.Timedelta(hours=1)

    T = int((end - start).total_seconds() // 60)
    _dbg(cfg, f"[DATA] Date range: {start} → {end} (minutes={T})")
    _dbg(cfg, f"[DATA] Lots in CSV: {sorted(df['parking_lot'].unique().tolist())}")
    _dbg(cfg, f"[DATA] Records total: {len(df)}")

    return max(T, 0)


def run_once(cfg, policy, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    stats = Stats()

    # --- Initialize lots ---
    lots = [Lot(**d) for d in cfg["lots"]]
    for L in lots:
        L.build(env)
        stats.arrivals_expected_by_lot[L.name] = 0
        stats.arrivals_spawned_by_lot[L.name]  = 0
        stats.served_by_lot[L.name]            = 0
        stats.rejected_by_lot[L.name]          = 0

        stats.price_sum[L.name] = 0.0
        stats.price_n[L.name] = 0

    # --- Optional dynamic pricing ---
    if hasattr(policy, "schedule"):
        env.process(policy.schedule(env, lots, stats))

    # --- Pick arrivals mode ---
    if cfg.get("arrival_mode") == "dataset":
        sample_duration = _make_sample_duration(cfg)
        arrivals_from_hourly_dataset(env, lots, policy, stats, cfg, sample_duration)
        T = _compute_horizon_minutes(cfg.get("dataset_path", "dataset_fluxos_hourly_2022.csv"), cfg)
    else:
        # Synthetic mode (kept for testing)
        dl = cfg.get("duration_lognorm") or {}
        m = float(dl.get("mean", 2.2))
        s = float(dl.get("sigma", 0.7))
        sample_duration = lambda: float(np.random.lognormal(m, s))
        env.process(arrivals_nhpp(env, lots, policy, stats, sample_duration, cfg.get("max_wait", 1), cfg))
        T = int(cfg.get("T", 1440))  # fallback: 1 day

    # --- Sampler ---
    env.process(sampler(env, lots, stats, step=5))

    # --- Run simulation ---
    env.run(until=T)
        # --- Consistency checks (debug) ---
    if cfg.get("debug", False):
        print("\n[CHECK] Arrivals (spawned) by lot:", stats.arrivals_spawned_by_lot)
        print("[CHECK] Expected (adj_entries) by lot:", stats.arrivals_expected_by_lot)
        print("[CHECK] Served by lot:", stats.served_by_lot)
        print("[CHECK] Rejected by lot:", stats.rejected_by_lot)
        print("[CHECK] Rejected by reason:", stats.rejected_reason)

        total_spawned = sum(stats.arrivals_spawned_by_lot.values())
        total_served  = sum(stats.served_by_lot.values())
        total_rej     = sum(stats.rejected_by_lot.values())

        print(f"[CHECK] totals -> spawned={total_spawned}  served={total_served}  rejected={total_rej}")
        if total_spawned != (total_served + total_rej):
            print("[WARN] spawned != served + rejected. Redirections and timeouts might explain gaps.")


    # Use T as the horizon in summaries
    return stats.to_dict(lots, T)
