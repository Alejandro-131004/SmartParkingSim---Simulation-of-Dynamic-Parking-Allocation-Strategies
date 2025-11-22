import pandas as pd

input_temp_file = "data/ME_2022_S1.csv"
output_daily_file = "temp_min_max_station1.csv"

df = pd.read_csv(input_temp_file, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# Filter for PARAMETRO == TEMP and NR_ESTACAO == 1
df = df[(df['parametro'].str.upper() == 'TEMP') & (df['nr_estacao'].astype(str) == '1')]
df['timestamp'] = pd.to_datetime(df['dtm_local'], errors='coerce')
df['dia'] = df['timestamp'].dt.date
df['temperatura'] = pd.to_numeric(df['valor'], errors='coerce')

# Optional: filter only plausible Celsius readings
df = df[(df['temperatura'] >= -20) & (df['temperatura'] <= 50)]

# Now aggregate per day, station 1 only
daily = df.groupby('dia')['temperatura'].agg(temp_min='min', temp_max='max').reset_index()

daily.to_csv(output_daily_file, index=False)
print(f"✅ Saved daily temperature for station 1 to {output_daily_file}")
