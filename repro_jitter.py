
import sys
import os
import pandas as pd
import numpy as np

# Adjust path to import src
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from digital_twin_sim.occupancy_forecast import OccupancyForecaster
except ImportError:
    # helper for running from root
    sys.path.append(os.path.join(os.getcwd(), 'src', 'digital_twin_sim'))
    from occupancy_forecast import OccupancyForecaster

def calculate_roughness(series):
    """
    Calculates a simple roughness metric: average absolute difference between consecutive points.
    Lower is smoother.
    """
    return np.mean(np.abs(np.diff(series)))

def test_forecast_roughness():
    # Mock training data
    dates = pd.date_range(start='2022-01-01', end='2022-01-07', freq='30min')
    data = {
        'date': dates.date,
        'hour': dates.hour + dates.minute/60.0,
        'occupancy': np.sin(np.linspace(0, 10, len(dates))) * 50 + 50, # smooth sine wave
        'entries': np.abs(np.sin(np.linspace(0, 10, len(dates)))) * 10,
        'exits': np.abs(np.cos(np.linspace(0, 10, len(dates)))) * 10,
        'traffic_score': np.ones(len(dates))
    }
    df_train = pd.DataFrame(data)
    df_train['date'] =  pd.to_datetime(df_train['date'])
    
    forecaster = OccupancyForecaster()
    forecaster.fit(df_train)
    
    # Predict
    test_start = '2022-01-08'
    test_end = '2022-01-08 12:00'
    
    print("Generating Stochastic Forecast...")
    df_pred = forecaster.predict(test_start, test_end, stochastic=True)
    
    roughness = calculate_roughness(df_pred['occupancy_pred'])
    print(f"Occupancy Roughness (Avg Abs Diff): {roughness:.4f}")
    
    roughness_entries = calculate_roughness(df_pred['entries_pred'])
    print(f"Entries Roughness (Avg Abs Diff): {roughness_entries:.4f}")

    return df_pred

if __name__ == "__main__":
    test_forecast_roughness()
