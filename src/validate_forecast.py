import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.dates as mdates

# Import your existing modules
from digital_twin_sim.occupancy_forecast import OccupancyForecaster

def main():
    # 1. Load Configuration
    config_path = os.path.join("config", "parking_P023.yaml")
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print("=== CONFIGURATION LOADED ===")
    print(f"Train Period: {cfg['twin_train_start']} to {cfg['twin_train_end']}")
    print(f"Test Period:  {cfg['twin_test_start']} to {cfg['twin_test_end']}")

    # 2. Load and Prepare Data
    dataset_path = cfg['dataset_path']
    if not os.path.exists(dataset_path):
        # Fallback if running from root and data is inside a folder not reflected in yaml relative path
        if os.path.exists(os.path.join("data", "P023_sim", "dataset_P023_simready_updated.csv")):
            dataset_path = os.path.join("data", "P023_sim", "dataset_P023_simready_updated.csv")
        else:
            print(f"[Error] Dataset not found: {dataset_path}")
            return

    print(f"[Data] Loading {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # If the maximum occupancy is > 1.5 (e.g., 50, 60, 100), we assume % and divide by 100.
    # This ensures that BOTH the training AND the test are on the correct scale (0.0 to 1.0)
    if 'occupancy' in df.columns and df['occupancy'].max() > 1.5:
        print("[Data] Scale fix: Converting occupancy from % (0-100) to ratio (0-1).")
        df['occupancy'] = df['occupancy'] / 100.0

    # Construct full datetime index
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df.set_index('datetime').sort_index()

    # 3. Split Train/Test
    train_start = pd.to_datetime(cfg['twin_train_start'])
    train_end = pd.to_datetime(cfg['twin_train_end']) + pd.Timedelta(hours=23, minutes=59)
    test_start = pd.to_datetime(cfg['twin_test_start'])
    test_end = pd.to_datetime(cfg['twin_test_end']) + pd.Timedelta(hours=23, minutes=59)

    df_train = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    
    # For testing, we need the real values to compare against
    df_test_real = df[(df.index >= test_start) & (df.index <= test_end)].copy()

    if df_train.empty or df_test_real.empty:
        print("[Error] Training or Test set is empty. Check dates in YAML.")
        return

    # 4. Train Forecaster
    print("[Model] Training OccupancyForecaster...")
    forecaster = OccupancyForecaster()
    # The forecaster expects columns: 'date', 'hour', 'occupancy', 'entries', 'exits', etc.
    # We pass the reset_index version or ensure columns exist
    df_train_reset = df_train.reset_index(drop=True)
    # Ensure date/hour columns are strings/ints as expected by fit()
    df_train_reset['date'] = df_train.index.date
    df_train_reset['hour'] = df_train.index.hour
    
    forecaster.fit(df_train_reset)

    # 5. Generate Prediction (Deterministic for validation)
    print("[Model] Generating Forecast...")
    # predict() returns a DataFrame with index at 5min intervals (or similar)
    df_pred = forecaster.predict(str(test_start), str(test_end), stochastic=False)

    # 6. Align Data for Comparison
    # The real data is hourly (at minute 0). The prediction is 5-min resolution.
    # We resample the prediction to hourly means (or take minute 0) to match ground truth.
    df_pred_hourly = df_pred.resample('1h').mean() # or .first()
    
    # Align indices
    common_index = df_pred_hourly.index.intersection(df_test_real.index)
    
    y_true = df_test_real.loc[common_index, 'occupancy']
    y_pred = df_pred_hourly.loc[common_index, 'occupancy_pred']

    # 7. Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R-squared manually or via sklearn
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\n=== VALIDATION RESULTS (Hourly Aggregation) ===")
    print(f"MAE  (Mean Absolute Error): {mae:.4f} (Occupancy Ratio)")
    print(f"RMSE (Root Mean Sq Error):  {rmse:.4f}")
    print(f"R² Score:                   {r2:.4f}")

    # 8. Visualization (FINAL SIMPLIFIED VERSION)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        y_true.index,
        y_true * 180,
        label='Ground Truth (Vehicles)',
        color='gray',
        alpha=0.8
    )
    ax.plot(
        y_pred.index,
        y_pred * 180,
        label='Forecast Model (Vehicles)',
        color='blue',
        linestyle='--',
        linewidth=1.5
    )

    ax.set_title(f"Occupancy Forecast Validation\nMAE: {mae:.3f} | R²: {r2:.3f}")
    ax.set_ylabel("Occupancy (Count)")
    ax.set_xlabel("Date")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- X-AXIS FIX (Limit trick) ---
    start_x = pd.to_datetime(cfg['twin_test_start'])

    # TRICK: End at 23:55 of day 30.
    # Since 01/07 starts at 00:00 of the next day, it falls outside the range
    # and the July tick label disappears.
    end_x = pd.to_datetime(cfg['twin_test_end']) + pd.Timedelta(hours=23, minutes=55)

    ax.set_xlim(start_x, end_x)

    # Use automatic locator (Matplotlib chooses optimal ticks: 01, 08, 15, ...)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # --------------------------------

    output_img = "reports/forecast_validation.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n[Plot] Saved validation plot to {output_img}")


if __name__ == "__main__":
    main()