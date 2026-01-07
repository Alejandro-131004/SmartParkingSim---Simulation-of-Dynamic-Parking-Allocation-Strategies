import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

class Forecaster:
    """
    Advanced forecasting model using Amazon Chronos (Pretrained Time Series Foundation Model).
    Replaces historical average with a token-based deep learning approach.
    """
    
    def __init__(self, dataset_path=None, df=None, model_name="amazon/chronos-t5-small"):
        self.dataset_path = dataset_path
        
        if df is not None:
            self.df = df.copy()
        elif dataset_path:
            self.df = pd.read_csv(dataset_path)
        else:
            raise ValueError("Must provide either dataset_path or df")
        
        # Standardize columns
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Create a proper datetime index for sorting
        # Assumes columns 'date' (YYYY-MM-DD) and 'hour' (0-23) exist
        if "date" in self.df.columns and "hour" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["date"]) + pd.to_timedelta(self.df["hour"], unit="h")
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        
        # Initialize Chronos Pipeline
        # device_map="cpu" ensures it runs everywhere; use "cuda" if you have a GPU
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",  
            torch_dtype=torch.float32,
        )

    def _get_context(self, column_name):
        """Helper to extract the historical time series context as a tensor."""
        # Clean data: drop NaNs which might break context
        series = self.df[column_name].dropna().values
        return torch.tensor(series, dtype=torch.float32)

    def predict_next_hours(self, horizon=3):
        """Forecasts occupancy for the next `horizon` hours."""
        context = self._get_context("occupancy")
        
        # Generate forecast (num_samples, prediction_length)
        forecast = self.pipeline.predict(
            context, 
            prediction_length=horizon,
            num_samples=20  # Sample 20 paths for robustness
        )
        
        # Take the median path as the point forecast
        # forecast shape: (1, num_samples, horizon) -> (num_samples, horizon)
        median_forecast = np.quantile(forecast[0].numpy(), 0.5, axis=0)
        
        return [float(x) for x in median_forecast]

    def predict_arrivals(self, horizon=3):
        """Forecasts entries (arrivals) for the next `horizon` hours."""
        context = self._get_context("entries")
        
        forecast = self.pipeline.predict(
            context, 
            prediction_length=horizon,
            num_samples=20
        )
        
        median_forecast = np.quantile(forecast[0].numpy(), 0.5, axis=0)
        
        # Arrivals must be integers and non-negative
        return [max(0, int(round(x))) for x in median_forecast]