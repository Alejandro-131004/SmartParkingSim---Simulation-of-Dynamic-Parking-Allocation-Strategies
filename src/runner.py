import yaml, json, statistics
from src.model import run_once
from src.policies import make_policy

def main():
    cfg = yaml.safe_load(open("config/base.yaml"))
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
    print("Served avg:", round(statistics.mean(served), 1))
    print("Arrivals avg:", round(statistics.mean(arrs), 1))

    # ocupação média por parque
    from collections import defaultdict
    price_sum, n = defaultdict(float), len(results)
    for r in results:
        for lot, v in r["avg_price"].items():
            price_sum[lot] += v
    print("Avg price:", {k: round(v/n, 2) for k, v in price_sum.items()})

    rej = [x["rejection_rate"] for x in results]
    rev = [x["revenue"] for x in results]
    wait = [x["wait_avg"] for x in results]
    print("Rejection rate avg:", round(statistics.mean(rej), 4))
    print("Revenue avg:", round(statistics.mean(rev), 2))
    print("Wait avg (min):", round(statistics.mean(wait), 2))
    # guardar resultados detalhados
    with open("reports/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
