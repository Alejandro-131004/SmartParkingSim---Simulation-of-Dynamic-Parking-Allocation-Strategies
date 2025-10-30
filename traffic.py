import pandas as pd

traffic_file = "trafegoall.csv"
output_file = "traffic_score_by_hour.csv"

df = pd.read_csv(traffic_file, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

df['fromto'] = pd.to_numeric(df['smo_fromto_total'], errors='coerce').fillna(0)
df['tofrom'] = pd.to_numeric(df['smo_tofrom_total'], errors='coerce').fillna(0)
df['volume_max'] = df[['fromto', 'tofrom']].max(axis=1)

df['timestamp'] = pd.to_datetime(df['dtm_local'], errors='coerce')
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# Identify the street global max (all-time max for each street)
street_max = df.groupby('local')['volume_max'].max()
df['global_street_max'] = df['local'].map(street_max)
df = df[df['global_street_max'] > 0]

# Score for each row
df['score'] = ((df['volume_max'] / df['global_street_max']) * 10).clip(0,10).round(0).fillna(0).astype(int)

# Group by street, date, hour, and coordinates (latitude/longitude are static per street)
grouped = df.groupby(['local','date','hour','latitude','longitude'], as_index=False)['score'].max()
grouped.to_csv(output_file, index=False)

print(f"✅ Saved traffic score per street per hour to {output_file}")
