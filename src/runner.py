import yaml, json, statistics
from model import run_once
from policies import make_policy
import os, pandas as pd
os.makedirs("reports", exist_ok=True)


def main():
    cfg = yaml.safe_load(open("config/base.yaml"))
    print("ev_share type:", type(cfg.get("ev_share")), "value:", cfg.get("ev_share"))
    policy = make_policy(cfg["policy"])
    results = []
    for r in range(cfg["runs"]): 
        seed = cfg["seed"] + r
        out = run_once(cfg, policy, seed)
        out["seed"] = seed
        results.append(out)
    # agregados simples
    served = [x["served"] for x in results]
    arrs   = [x["arrivals"] for x in results]
    pd.DataFrame(results).to_csv("reports/results.csv", index=False)
    print("Served avg:", round(statistics.mean(served), 1))
    print("Arrivals avg:", round(statistics.mean(arrs), 1))
    occg = [x["occ_global"] for x in results]
    print("Occ global avg:", round(statistics.mean(occg), 3))


    # average occupancy per parking lot
    from collections import defaultdict
    price_sum = defaultdict(float)
    n = len(results)
    for rec in results:
        for lot, v in rec["avg_price"].items():
            price_sum[lot] += v
    print("Avg price:", {k: round(v/n, 2) for k, v in price_sum.items()})


    rej = [x["rejection_rate"] for x in results]
    rev = [x["revenue_total"] for x in results]
    wait = [x["wait_avg"] for x in results]
    print("Rejection rate avg:", round(statistics.mean(rej), 4))
    print("Revenue avg:", round(statistics.mean(rev), 2))
    print("Wait avg (min):", round(statistics.mean(wait), 2))
    rpc = [x["revenue_per_car"] for x in results]
    print("Revenue per car avg:", round(statistics.mean(rpc), 2))
    energy = [x.get("energy_total", 0) for x in results]
    print("Energy delivered avg (kWh):", round(statistics.mean(energy), 2))
    evs = [x.get("ev_served", 0) for x in results]
    print("EV served avg:", round(statistics.mean(evs), 1))


    # guardar resultados detalhados
    with open("reports/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
