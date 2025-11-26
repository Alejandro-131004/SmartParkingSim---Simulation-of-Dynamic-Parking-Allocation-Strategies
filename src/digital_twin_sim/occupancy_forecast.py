import pandas as pd
import numpy as np


class OccupancyForecaster:
    """
    Baseline forecaster for parking occupancy.

    It learns:
        mean_occ[hour] = average occupancy for each hour-of-day (0..23)
    based on a training window defined by [train_start, train_end].

    Then, it predicts occupancy for the test window by:
        y_pred(t) = mean_occ[hour_of_day(t)]
    """

    def __init__(self):
        self.hour_means = None

    # ----------------------------------------------------------
    # Train model on df_train
    # ----------------------------------------------------------
    def fit(self, df_train):
        df = df_train.copy()
        df = df.sort_values(["date", "hour"])

        # Compute mean occupancy per hour of day
        hour_means = df.groupby("hour")["occupancy"].mean()

        # fill missing hours 0–23
        hour_means = hour_means.reindex(range(24))

        # forward/backward fill
        hour_means = hour_means.ffill().bfill()

        self.hour_means = hour_means

    # ----------------------------------------------------------
    # Predict occupancy on df_test
    # ----------------------------------------------------------
    def predict(self, df_test):
        df = df_test.copy()
        df = df.sort_values(["date", "hour"])

        if self.hour_means is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # forecast = hour_mean(hour_of_row)
        df["forecast_occ"] = df["hour"].apply(lambda h: self.hour_means.loc[int(h)])

        return df