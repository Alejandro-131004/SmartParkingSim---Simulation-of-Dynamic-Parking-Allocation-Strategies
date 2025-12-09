import pandas as pd
import numpy as np

class OccupancyForecaster:
    """
    Advanced Statistical Forecaster.
    Learns profiles for: Occupancy, Entries, Exits, AND Traffic Score.
    Resolution: 30 minutes.
    """

    def __init__(self):
        # Dictionary to store stats for all metrics
        self.stats = {}
        self.traffic_base_avg = 0.0 # Global average for normalization

    def fit(self, df_train):
        df = df_train.copy()
        
        # 1. Ensure correct datetime
        df['date_time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        df = df.set_index('date_time').sort_index()
        
        # 2. Resample to 30 minutes
        df_30min = pd.DataFrame()
        df_30min['occupancy'] = df['occupancy'].resample('30min').interpolate(method='linear')
        
        # Flows are summed/split (Hourly / 2 for 30min approx)
        df_30min['entries'] = df['entries'].resample('30min').ffill() / 2.0
        df_30min['exits'] = df['exits'].resample('30min').ffill() / 2.0
        
        # Traffic score is a state (mean)
        if 'traffic_score' in df.columns:
            df_30min['traffic'] = df['traffic_score'].resample('30min').interpolate(method='linear')
            self.traffic_base_avg = df['traffic_score'].mean() # Store global mean
        else:
            # Fallback if column missing
            df_30min['traffic'] = 1.0
            self.traffic_base_avg = 1.0

        # 3. Auxiliary features
        df_30min['minute_of_day'] = df_30min.index.hour * 60 + df_30min.index.minute
        df_30min['is_weekend'] = df_30min.index.dayofweek >= 5 
        
        # 4. Calculate stats (Mean and Std Dev)
        metrics = ['occupancy', 'entries', 'exits', 'traffic']
        
        self.stats = df_30min.groupby(['is_weekend', 'minute_of_day'])[metrics].agg(['mean', 'std'])
        self.stats = self.stats.fillna(0.0)
        
        print(f"[Forecaster] Model fitted. Profiles learned for: {metrics}")
        print(f"[Forecaster] Global Average Traffic Score: {self.traffic_base_avg:.2f}")

    def predict(self, start_date, end_date, stochastic=False):
        if self.stats.empty:
            raise ValueError("Model not trained.")
            
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        # Prepare output structure
        cols = ['occupancy_pred', 'entries_pred', 'exits_pred', 'traffic_pred']
        data = {c: [] for c in cols}
        
        for ts in dates:
            minute_of_day = ts.hour * 60 + ts.minute
            lookup_minute = (minute_of_day // 30) * 30
            is_weekend = ts.dayofweek >= 5
            
            try:
                slot_stats = self.stats.loc[(is_weekend, lookup_minute)]
                
                for metric, col_name in zip(['occupancy', 'entries', 'exits', 'traffic'], cols):
                    mu = slot_stats[metric]['mean']
                    sigma = slot_stats[metric]['std']
                    
                    # Scale flows to 5-min
                    if metric in ['entries', 'exits']:
                        mu /= 6.0
                        sigma /= np.sqrt(6.0)
                    
                    if stochastic:
                        val = np.random.normal(mu, sigma)
                    else:
                        val = mu
                    
                    # Constraints
                    val = max(0.0, val)
                    data[col_name].append(val)
                
            except KeyError:
                for c in cols: data[c].append(0.0)
            
        return pd.DataFrame(data, index=dates)