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
        
        # --- SCALE FIX (KEEP THIS) ---
        if 'occupancy' in df.columns:
            max_val = df['occupancy'].max()
            if max_val > 1.5:
                print(f"[Forecaster] DETECTED SCALE ISSUE: Max occupancy is {max_val}. Converting % to ratio (dividing by 100).")
                df['occupancy'] = df['occupancy'] / 100.0
        # -----------------------------

        if 'date_time' not in df.columns:
            df['date_time'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        
        df = df.set_index('date_time').sort_index()
        
        # Resample to 30 minutes
        df_30min = pd.DataFrame()
        df_30min['occupancy'] = df['occupancy'].resample('30min').interpolate(method='linear')
        df_30min['entries'] = df['entries'].resample('30min').ffill() / 2.0
        df_30min['exits'] = df['exits'].resample('30min').ffill() / 2.0
        
        if 'temp_min_c' in df.columns:
            df_30min['temp_min_c'] = df['temp_min_c'].resample('30min').ffill()
            df_30min['temp_max_c'] = df['temp_max_c'].resample('30min').ffill()
        
        if 'traffic_score' in df.columns:
            df_30min['traffic'] = df['traffic_score'].resample('30min').interpolate(method='linear')
            self.traffic_base_avg = df['traffic_score'].mean()
        else:
            df_30min['traffic'] = 1.0
            self.traffic_base_avg = 1.0

        # --- MAJOR CHANGE HERE ---
        # Instead of just 'is_weekend', we use 'day_of_week' (0=Mon, 6=Sun)
        df_30min['minute_of_day'] = df_30min.index.hour * 60 + df_30min.index.minute
        df_30min['day_of_week'] = df_30min.index.dayofweek 
        
        metrics = ['occupancy', 'entries', 'exits', 'traffic', 'temp_min_c', 'temp_max_c']
        
        # Group by DAY_OF_WEEK and MINUTE
        self.stats = df_30min.groupby(['day_of_week', 'minute_of_day'])[metrics].agg(['mean', 'std'])
        self.stats = self.stats.fillna(0.0)
        
        print(f"[Forecaster] Model fitted (Granular). Profiles learned for: {metrics} by Day of Week.")


    def _generate_correlated_noise(self, n_steps, sigma, phi=0.9):
        """
        Generates a correlated noise sequence (AR(1) process).
        x_t = phi * x_{t-1} + (1 - phi) * white_noise
        Scales the result to have approximately 'sigma' standard deviation.
        """
        white_noise = np.random.normal(0, 1, n_steps)
        noise = np.zeros(n_steps)
        # Initialize
        noise[0] = white_noise[0]
        for t in range(1, n_steps):
            noise[t] = phi * noise[t-1] + np.sqrt(1 - phi**2) * white_noise[t]
        
        return noise * sigma

    def predict(self, start_date, end_date, stochastic=False):
        if self.stats.empty:
            raise ValueError("Model not trained.")
            
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        n_steps = len(dates)
        
        cols = ['occupancy_pred', 'entries_pred', 'exits_pred', 'traffic_pred', 'temp_min_c', 'temp_max_c']
        data = {c: np.zeros(n_steps) for c in cols}
        
        noise_vectors = {}
        if stochastic:
            for col in cols:
                phi = 0.95 if 'occupancy' in col else 0.8
                noise_vectors[col] = self._generate_correlated_noise(n_steps, sigma=1.0, phi=phi)
        
        for i, ts in enumerate(dates):
            minute_of_day = ts.hour * 60 + ts.minute
            lookup_minute = (minute_of_day // 30) * 30
            
            # --- MAJOR CHANGE HERE ---
            # Lookup based on specific day of week (0-6)
            day_of_week = ts.dayofweek
            
            try:
                # Use day_of_week instead of is_weekend
                slot_stats = self.stats.loc[(day_of_week, lookup_minute)]
                
                for metric, col_name in zip(['occupancy', 'entries', 'exits', 'traffic', 'temp_min_c', 'temp_max_c'], cols):
                    mu = slot_stats[metric]['mean']
                    sigma = slot_stats[metric]['std']
                    
                    if metric in ['entries', 'exits']:
                        mu /= 6.0
                        sigma /= np.sqrt(6.0)
                    
                    val = mu
                    if stochastic:
                        if sigma > 0:
                            noise_val = noise_vectors[col_name][i]
                            val += noise_val * sigma
                    
                    val = max(0.0, val)
                    data[col_name][i] = val
                
            except KeyError:
                pass 
            
        df_res = pd.DataFrame(data, index=dates)
        df_res['avg_temp'] = (df_res['temp_min_c'] + df_res['temp_max_c']) / 2.0
        return df_res