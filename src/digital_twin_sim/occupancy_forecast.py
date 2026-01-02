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
        
        # Temp is state (mean)
        if 'temp_min_c' in df.columns:
            df_30min['temp_min_c'] = df['temp_min_c'].resample('30min').ffill()
            df_30min['temp_max_c'] = df['temp_max_c'].resample('30min').ffill()
        
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
        metrics = ['occupancy', 'entries', 'exits', 'traffic', 'temp_min_c', 'temp_max_c']
        
        self.stats = df_30min.groupby(['is_weekend', 'minute_of_day'])[metrics].agg(['mean', 'std'])
        self.stats = self.stats.fillna(0.0)
        
        print(f"[Forecaster] Model fitted. Profiles learned for: {metrics}")
        print(f"[Forecaster] Global Average Traffic Score: {self.traffic_base_avg:.2f}")

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
        
        # Prepare output structure
        cols = ['occupancy_pred', 'entries_pred', 'exits_pred', 'traffic_pred', 'temp_min_c', 'temp_max_c']
        data = {c: np.zeros(n_steps) for c in cols}
        
        # Pre-generate noise if stochastic
        noise_vectors = {}
        if stochastic:
            # We use a fixed sigma estimate for the noise generation to simplify
            # Or we could just generate N(0,1) correlated noise and scale it later by local sigma.
            # Let's generate unit variance correlated noise and scale by local sigma.
            for col in cols:
                # phi=0.95 gives very smooth, slow-moving noise (good for occupancy)
                # phi=0.7 gives bit more jagged noise (good for flow)
                phi = 0.95 if 'occupancy' in col else 0.8
                noise_vectors[col] = self._generate_correlated_noise(n_steps, sigma=1.0, phi=phi)
        
        for i, ts in enumerate(dates):
            minute_of_day = ts.hour * 60 + ts.minute
            lookup_minute = (minute_of_day // 30) * 30
            is_weekend = ts.dayofweek >= 5
            
            try:
                slot_stats = self.stats.loc[(is_weekend, lookup_minute)]
                
                for metric, col_name in zip(['occupancy', 'entries', 'exits', 'traffic', 'temp_min_c', 'temp_max_c'], cols):
                    mu = slot_stats[metric]['mean']
                    sigma = slot_stats[metric]['std']
                    
                    # Scale flows to 5-min
                    if metric in ['entries', 'exits']:
                        mu /= 6.0
                        sigma /= np.sqrt(6.0)
                    
                    val = mu
                    if stochastic:
                        # Apply pre-generated correlated noise scaled by local sigma
                        if sigma > 0:
                            noise_val = noise_vectors[col_name][i]
                            val += noise_val * sigma
                        else:
                            # If sigma is 0, add small jitter to avoid flat lines if requested
                            pass

                    # Constraints
                    val = max(0.0, val)
                    data[col_name][i] = val
                
            except KeyError:
                pass # Default is 0.0
            
        df_res = pd.DataFrame(data, index=dates)
        df_res['avg_temp'] = (df_res['temp_min_c'] + df_res['temp_max_c']) / 2.0
        return df_res