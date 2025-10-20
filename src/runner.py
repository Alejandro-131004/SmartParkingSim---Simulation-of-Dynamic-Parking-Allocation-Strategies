import yaml, json, statistics
from model import run_once
from policies import make_policy
import os, pandas as pd
os.makedirs("reports", exist_ok=True)


def main():
    # --- Load configuration ---
    cfg = yaml.safe_load(open("config/base.yaml"))
    print("ev_share type:", type(cfg.get("ev_share")), "value:", cfg.get("ev_share"))

    # --- Create policy object ---
    policy = make_policy(cfg["policy"])

    # --- NEW: Attach elasticities and event calendar ---
    if hasattr(policy, "set_elasticities"):
        policy.set_elasticities(cfg.get("elasticity", {}))
    if hasattr(policy, "set_event_calendar"):
        policy.set_event_calendar(cfg.get("event_calendar", []))

    # --- Run multiple simulations ---
    results = []
    for r in range(cfg["runs"]): 
        seed = cfg["seed"] + r
        out = run_once(cfg, policy, seed)
        out["seed"] = seed
        results.append(out)

    # --- Aggregated statistics ---
    served = [x["served"] for x in results]
    arrs   = [x["arrivals"] for x in results]
    pd.DataFrame(results).to_csv("reports/results.csv", index=False)
    print("Served avg:", round(statistics.mean(served), 1))
    print("Arrivals avg:", round(statistics.mean(arrs), 1))

    occg = [x["occ_global"] for x in results]
    print("Occ global avg:", round(statistics.mean(occg), 3))

    # --- Average occupancy per parking lot ---
    from collections import defaultdict
    price_sum = defaultdict(float)
    n = len(results)
    for rec in results:
        for lot, v in rec["avg_price"].items():
            price_sum[lot] += v
    print("Avg price:", {k: round(v/n, 2) for k, v in price_sum.items()})

    # --- Key metrics ---
    rej = [x["rejection_rate"] for x in results]
    rev = [x["revenue_total"] for x in results]
    wait = [x["wait_avg"] for x in results]
    rpc = [x["revenue_per_car"] for x in results]
    energy = [x.get("energy_total", 0) for x in results]
    evs = [x.get("ev_served", 0) for x in results]
    rej_ev = [x["rej_no_ev"] for x in results]
    rej_spot = [x["rej_no_spot"] for x in results]
    redir = [x.get("redirected", 0) for x in results]
    redir_ok = [x.get("redirect_success", 0) for x in results]

    print("Rejection rate avg:", round(statistics.mean(rej), 4))
    print("Revenue avg:", round(statistics.mean(rev), 2))
    print("Revenue per car avg:", round(statistics.mean(rpc), 2))
    print("Wait avg (min):", round(statistics.mean(wait), 2))
    print("Energy delivered avg (kWh):", round(statistics.mean(energy), 2))
    print("EV served avg:", round(statistics.mean(evs), 1))
    print("Rejections (no spot / no EV):", round(statistics.mean(rej_spot),1), "/", round(statistics.mean(rej_ev),1))
    print("Redirections avg:", round(statistics.mean(redir), 1))
    print("Successful redirections avg:", round(statistics.mean(redir_ok), 1))

    # --- Save detailed results ---
    with open("reports/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to:")
    print(" - reports/results.csv")
    print(" - reports/results.json")


if __name__ == "__main__":
    main()
