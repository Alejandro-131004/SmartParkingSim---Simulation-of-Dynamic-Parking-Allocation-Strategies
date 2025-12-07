import pandas as pd
import numpy as np

class OccupancyForecaster:
    """
    Statistical Forecaster (Mean + Std Dev) with 30-minute resolution.
    """

    def __init__(self):
        self.profile_stats = None

    def fit(self, df_train):
        df = df_train.copy()
        
        # 1. Ensure correct datetime
        df['date_time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        df = df.set_index('date_time').sort_index()
        
        # 2. Resample to 30 minutes (FIX: '30min' instead of '30T')
        df_30min = df['occupancy'].resample('30min').interpolate(method='linear').to_frame()
        
        # 3. Auxiliary features
        df_30min['minute_of_day'] = df_30min.index.hour * 60 + df_30min.index.minute
        df_30min['is_weekend'] = df_30min.index.dayofweek >= 5 
        
        # 4. Calculate stats (Mean and Std Dev)
        self.profile_stats = df_30min.groupby(['is_weekend', 'minute_of_day'])['occupancy'].agg(['mean', 'std'])
        self.profile_stats['std'] = self.profile_stats['std'].fillna(0.0)
        
        print(f"[Forecaster] Model fitted. Profiles learned: {len(self.profile_stats)}")

    def predict(self, start_date, end_date, stochastic=False):
        if self.profile_stats is None:
            raise ValueError("Model not trained.")
            
        # 5-minute timeline (FIX: '5min' instead of '5T')
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        preds = []
        
        for ts in dates:
            minute_of_day = ts.hour * 60 + ts.minute
            lookup_minute = (minute_of_day // 30) * 30
            is_weekend = ts.dayofweek >= 5
            
            try:
                stats = self.profile_stats.loc[(is_weekend, lookup_minute)]
                mu = stats['mean']
                sigma = stats['std']
                
                if stochastic:
                    val = np.random.normal(mu, sigma)
                else:
                    val = mu
                val = max(0.0, val)
                
            except KeyError:
                val = 0.0
            
            preds.append(val)
            
        return pd.DataFrame({'occupancy_pred': preds}, index=dates)