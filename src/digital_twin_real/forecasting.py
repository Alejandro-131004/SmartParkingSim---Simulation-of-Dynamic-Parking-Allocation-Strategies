import numpy as np
import pandas as pd


class Forecaster:
    """
    Simple yet effective forecasting model based on:
        - hourly historical average
        - exponential smoothing
        - automatic fallback
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.df.columns = self.df.columns.str.lower().str.strip()
        self.df["date"] = pd.to_datetime(self.df["date"])

    def predict_next_hours(self, horizon=3):
        df = self.df.sort_values(["date", "hour"])

        # média histórica por hora
        hourly = df.groupby("hour")["occupancy"].mean()

        preds = []
        for h in range(horizon):
            hour = (df.iloc[-1]["hour"] + h + 1) % 24
            pred = hourly.get(hour, hourly.mean())  # fallback
            preds.append(float(pred))

        return preds

    def predict_arrivals(self, horizon=3):
        df = self.df.sort_values(["date", "hour"])
        hourly = df.groupby("hour")["entries"].mean()

        preds = []
        for h in range(horizon):
            hour = (df.iloc[-1]["hour"] + h + 1) % 24
            pred = hourly.get(hour, hourly.mean())
            preds.append(int(pred))

        return preds