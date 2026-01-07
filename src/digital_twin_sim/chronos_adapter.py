
import pandas as pd
import numpy as np
import sys
import os
import tqdm

# Ensure we can import from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from digital_twin_sim.occupancy_forecast import OccupancyForecaster
from digital_twin_real.forecasting import Forecaster as ChronosModel

class ChronosAdapter(OccupancyForecaster):
    """
    Adapter that combines the Statistical Forecaster (for aux variables like traffic/weather)
    with the Deep Learning Amazon Chronos Forecaster (for occupancy/entries).
    
    Implements Rolling Horizon Forecast (Hourly updates).
    """

    def __init__(self):
        super().__init__()
        self.chronos = None
        self.last_train_df = None

    def _prepare_hourly_df(self, df_raw):
        """Helper to convert raw 5-min/mixed simulation data to cleanly formatted Hourly data for Chronos."""
        df = df_raw.copy()
        
        # Construct Datetime Index if missing
        if 'date_time' not in df.columns:
            # Try standard sim columns
            if 'date' in df.columns and 'hour' in df.columns:
                 # If 'minute' is present, use it, otherwise assume 0
                 df['minute'] = df['minute'] if 'minute' in df.columns else 0
                 df['date_time'] = pd.to_datetime(df['date']) + \
                                   pd.to_timedelta(df['hour'], unit='h') + \
                                   pd.to_timedelta(df['minute'], unit='m')
            elif 'date' in df.columns and 'time_min' in df.columns:
                 df['date_time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time_min'], unit='m')
            else:
                 # Fallback: assume index is datetime if not implicit
                 if not isinstance(df.index, pd.DatetimeIndex):
                     raise ValueError("Cannot parse dataframe datetime. Missing date/hour columns.")
                 df['date_time'] = df.index

        df = df.set_index('date_time').sort_index()

        # Resample to Hourly
        # Aggregation: 
        # - Usage/Occupancy -> MEAN is best (average occ over the hour)
        # - Entries -> SUM (total entries in the hour)
        # - Exits -> SUM
        
        agg_dict = {
            'occupancy': 'mean',
            'entries': 'sum',
            'exits': 'sum' 
        }
        # Handle missing columns gracefully
        agg_valid = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        df_hourly = df.resample('1h').agg(agg_valid)
        
        # Ensure we have required columns for Chronos (it uses 'occupancy' and 'entries' by name)
        # Chronos Wrapper expects lower case columns.
        
        return df_hourly.reset_index() # Return with date_time as column or index? 
        # Forecaster expects 'date' and 'hour' columns OR a clean DF where it can find 'occupancy'.
        # Actually Forecaster init just needs 'occupancy' column. 
        # It creates 'timestamp' from 'date'+'hour' IF they exist, but if we pass a DF, it just needs the values for context.
        # But wait, Forecaster.__init__ line 27:
        # if "date" in self.df.columns and "hour" in self.df.columns: self.df["timestamp"]...
        # It sorts by timestamp.
        # If I pass a DF that has 'occupancy' but not date/hour, it might maintain order (if already sorted).
        # To be safe, let's ensure it has date/hour.
        
        df_hourly['date'] = df_hourly['date_time'].dt.date
        df_hourly['hour'] = df_hourly['date_time'].dt.hour
        return df_hourly

    def fit(self, df_train):
        # 1. Fit the statistical model (to get traffic/weather profiles)
        super().fit(df_train)
        
        # 2. Prepare Hourly Data for Chronos
        print("[ChronosAdapter] Resampling training data to HOURLY resolution for Chronos...")
        self.last_train_df = self._prepare_hourly_df(df_train)
        
        # 3. Initialize Chronos
        print(f"[ChronosAdapter] Initializing Amazon Chronos Model with {len(self.last_train_df)} hourly samples...")
        self.chronos = ChronosModel(df=self.last_train_df)
        print("[ChronosAdapter] Chronos initialized successfully.")

    def predict(self, start_date, end_date, stochastic=False, ground_truth_df=None):
        # 1. Get baseline structure from Statistical model (fills timestamps, weather, traffic)
        print("[ChronosAdapter] Generating baseline structure...")
        df_res = super().predict(start_date, end_date, stochastic=stochastic)
        
        # 2. HOURLY GRID
        # We need to predict for every hour in the range [start, end]
        t_start = pd.to_datetime(start_date)
        t_end = pd.to_datetime(end_date)
        
        # Align to hour start
        t_start_h = t_start.floor('h')
        t_end_h = t_end.ceil('h')
        
        # Hours to forecast
        pred_dates_hourly = pd.date_range(start=t_start_h, end=t_end_h, freq='h')
        total_hours = len(pred_dates_hourly) - 1 # Interval count
        if total_hours < 1: total_hours = 1
        
        print(f"[ChronosAdapter] Starting ROLLING FORECAST for {total_hours} hours. This may take time...")

        if ground_truth_df is None:
            print("[WARN] No Ground Truth provided for Rolling Forecast! Falling back to Iterative Prediction (auto-regressive). Accuracy will degrade.")
            # If no ground truth, we can only feed back our own predictions? 
            # Or just run plain horizon?
            # User specifically asked for "updated info". 
            # We'll default to standard horizon prediction if GT missing.
            occ_hourly = self.chronos.predict_next_hours(total_hours)
            ent_hourly = self.chronos.predict_arrivals(total_hours)
        else:
            # --- ROLLING FORECAST LOOP ---
            
            # Prepare GT Hourly
            gt_hourly = self._prepare_hourly_df(ground_truth_df)
            # Ensure GT is indexed by time for fast lookup
            gt_hourly_idx = gt_hourly.set_index('date_time')
            
            occ_preds = []
            ent_preds = []
            
            # Initial context is set in self.chronos (from training)
            
            # We iterate through each target hour.
            # To predict Hour H, we need context up to H-1.
            # After predicting H, we reveal H (Actual) to context before predicting H+1.
            
            # Note: Chronos wrapper simply takes `self.df`. We must append to it.
            # Work on a copy of the model/df to avoid polluting if we re-run? 
            # The simulator creates a new instance each run, so safe.
            
            # Optim: Chronos pipeline context extraction is O(N). Appending O(1).
            # The forecast loop:
            
            for i in range(total_hours):
                current_target_time = pred_dates_hourly[i]
                
                # 1. Predict Next Step (1 Hour)
                # Note: predict_next_hours(1) uses current self.df as context
                p_occ = self.chronos.predict_next_hours(1)[0]
                p_ent = self.chronos.predict_arrivals(1)[0]
                
                occ_preds.append(p_occ)
                ent_preds.append(p_ent)
                
                # 2. UPDATE CONTEXT with ACTUAL data for this timestep
                # We need the ACTUAL data for 'current_target_time' to be added to history
                # so that the prediction for (i+1) uses it.
                
                # Find the actual row in GT
                try:
                    # Look for the hour we just predicted
                    actual_row = gt_hourly_idx.loc[current_target_time]
                    
                    # Construct row to append
                    # We need 'occupancy' and 'entries' columns primarily
                    new_row = pd.DataFrame([{
                        'occupancy': actual_row['occupancy'],
                        'entries': actual_row['entries'],
                        # Add date/hour if needed by implementation (Forecaster uses values directly usually)
                        'date': str(current_target_time.date()),
                        'hour': current_target_time.hour
                    }])
                    
                    # Append to internal DF
                    self.chronos.df = pd.concat([self.chronos.df, new_row], ignore_index=True)
                    
                except KeyError:
                    # If GT missing for this hour, use Prediction (Auto-regressive fallback)
                    # print(f"[Warn] Missing GT for {current_target_time}. Using prediction.")
                    new_row = pd.DataFrame([{
                        'occupancy': p_occ,
                        'entries': p_ent
                    }])
                    self.chronos.df = pd.concat([self.chronos.df, new_row], ignore_index=True)
                
                # Progress logging
                if i % 24 == 0:
                     sys.stdout.write(f"\r[ChronosAdapter] Progress: {i}/{total_hours} hours...")
                     sys.stdout.flush()

            print("") # Newline
            occ_hourly = occ_preds
            ent_hourly = ent_preds


        # 3. INTERPOLATION (Hourly -> 5min)
        # Create DF
        # Be careful with length matching
        # pred_dates_hourly has len = total_hours + 1? No.
        # If we predicated `total_hours` points, we need corresponding timestamps.
        # We assume predictions start from t_start
        
        # We used pred_dates_hourly[0] as first target.
        ts_index = pred_dates_hourly[:len(occ_hourly)]
        
        df_chronos = pd.DataFrame({
            'occupancy_cnt': occ_hourly,
            'entries_cnt': ent_hourly
        }, index=ts_index)
        
        # Reindex to simulation 5-min steps
        df_chronos_5min = df_chronos.reindex(df_res.index)
        
        # Interpolate
        df_chronos_5min['occupancy_cnt'] = df_chronos_5min['occupancy_cnt'].interpolate(method='time')
        
        # Entries: Hourly count -> 5 min count. 
        # Divide by 12.0
        df_chronos_5min['entries_cnt'] = df_chronos_5min['entries_cnt'].fillna(method='ffill') / 12.0
        
        # Fill edges
        df_chronos_5min = df_chronos_5min.ffill().bfill()
        
        # 4. MERGE
        common_idx = df_res.index.intersection(df_chronos_5min.index)
        
        df_res.loc[common_idx, 'occupancy_pred'] = df_chronos_5min.loc[common_idx, 'occupancy_cnt']
        df_res.loc[common_idx, 'entries_pred'] = df_chronos_5min.loc[common_idx, 'entries_cnt']
        
        # Clamp
        df_res['occupancy_pred'] = df_res['occupancy_pred'].clip(lower=0.0)
        df_res['entries_pred'] = df_res['entries_pred'].clip(lower=0.0)
        
        print("[ChronosAdapter] Rolling Forecast merged.")
        return df_res
