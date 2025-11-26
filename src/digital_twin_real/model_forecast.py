import random

from model import driver


def arrivals_from_forecast(
    env,
    lots,
    forecast_arrivals,
    policy,
    stats,
    sample_duration,
    max_wait,
    cfg,
):
    """
    Generate arrivals for the next forecasted hours based ONLY on the forecast.

    forecast_arrivals: list of predicted arrivals per hour, e.g. [40, 36, 29].

    For each hour h:
      - We create forecast_arrivals[h] drivers.
      - Arrivals are spread uniformly across the 60 minutes of that hour.
      - Each driver uses the same 'driver()' process as the historical mode,
        so all pricing, EV logic, redirections, etc. remain unchanged.
    """
    car_id = 0

    for hour_idx, arrivals_this_hour in enumerate(forecast_arrivals):
        n_in = max(0, int(arrivals_this_hour))

        if n_in > 0:
            interval = 60.0 / n_in
        else:
            interval = 60.0  # no arrivals this hour, just advance one hour

        for _ in range(n_in):
            # In your configuration for P023 there is a single lot.
            lot = lots[0]

            stats.arrivals += 1
            stats.arrivals_spawned_by_lot[lot.name] = (
                stats.arrivals_spawned_by_lot.get(lot.name, 0) + 1
            )

            car_name = f"F{hour_idx}_{car_id}"
            car_id += 1

            env.process(
                driver(
                    env=env,
                    i=car_name,
                    lots=[lot],
                    policy=policy,
                    stats=stats,
                    sample_duration=sample_duration,
                    max_wait=max_wait,
                    cfg=cfg,
                )
            )

            yield env.timeout(interval)

        # After this loop, we have advanced ~60 minutes due to the intervals.
