
import pandas as pd
import os

def inspect_data():
    path = "data/P023_sim/dataset_P023_simready_updated.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Try to construct datetime
    try:
        if 'date' in df.columns and 'hour' in df.columns:
            df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
            df = df.set_index('datetime').sort_index()
        else:
            print("Missing 'date' or 'hour' columns.")
            print(df.head())
            return
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return

    print("\n--- Data Range ---")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")
    print(f"Total Rows: {len(df)}")
    
    # Check for gaps
    print("\n--- Time Diff Check ---")
    time_diffs = df.index.to_series().diff().dropna()
    print(time_diffs.value_counts().head())
    
    max_gap = time_diffs.max()
    print(f"Max gap: {max_gap}")

    if max_gap > pd.Timedelta(hours=2):
        print("\n[WARNING] Large gaps detected!")
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
        print(gaps)

    # Check for "stuck" values (3 months full/empty)
    print("\n--- Value Distribution (Occupancy) ---")
    print(df['occupancy'].describe())
    
    print("\n--- Check for Consecutive Identical Values (Frozen Data) ---")
    # Identify blocks where occupancy doesn't change for > 24 hours
    df['occ_change'] = df['occupancy'].diff().ne(0)
    df['block_id'] = df['occ_change'].cumsum()
    
    blocks = df.groupby('block_id')['occupancy'].agg(['first', 'count'])
    # Assuming hourly data, 24 count = 1 day
    long_blocks = blocks[blocks['count'] > 24]
    
    if not long_blocks.empty:
        print(f"\n[found] {len(long_blocks)} blocks where occupancy is frozen for > 24h.")
        print(long_blocks.sort_values('count', ascending=False).head(10))
    else:
        print("No >24h frozen blocks found.")

if __name__ == "__main__":
    inspect_data()
