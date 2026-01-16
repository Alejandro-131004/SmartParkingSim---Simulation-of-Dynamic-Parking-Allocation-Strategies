import pandas as pd
import numpy as np
import ast
from concurrent.futures import ProcessPoolExecutor
import sys

def extract_coordinates(coord_str):
    try:
        if isinstance(coord_str, str):
            coord_dict = ast.literal_eval(coord_str)
        else:
            coord_dict = coord_str
        if isinstance(coord_dict, dict) and 'coordinates' in coord_dict:
            lon, lat = coord_dict['coordinates']
            return lat, lon
    except Exception:
        pass
    return None, None

def get_weighted_traffic_score(ts, lat, lon, traffic_df, top_n=5, max_dist_km=2.0):
    # Ensure date/hour matching works
    target_date = pd.to_datetime(ts).date()
    target_hour = pd.to_datetime(ts).hour
    
    traffic_hour = traffic_df[
        (traffic_df['date'] == target_date) & (traffic_df['hour'] == target_hour)
    ]
    
    if traffic_hour.empty:
        return np.nan

    rads = np.radians
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371 # Earth radius
        dlat = rads(lat2-lat1)
        dlon = rads(lon2-lon1)
        a = np.sin(dlat/2)**2 + np.cos(rads(lat1)) * np.cos(rads(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    traffic_hour = traffic_hour.copy()
    traffic_hour['distance_km'] = traffic_hour.apply(
        lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1
    )
    
    traffic_near = traffic_hour[traffic_hour['distance_km'] <= max_dist_km].sort_values('distance_km').head(top_n)
    
    if not traffic_near.empty:
        weights = 1 / (traffic_near['distance_km'].replace(0, 0.1)**2)
        score = np.average(traffic_near['score'], weights=weights)
        return round(float(score), 2)
    return np.nan

def traffic_score_row(args):
    row, traffic_df = args
    return get_weighted_traffic_score(row['timestamp_hour'], row['lat'], row['lon'], traffic_df)

def preprocess_parking_data_simplified(input_file, output_file, traffic_score_file, nthreads=8):
    print("\n" + "="*70)
    print(" PRE-PROCESSAMENTO: OCUPAÇÃO E TRÁFEGO")
    print("="*70)

    # 1. Loading and cleaning headers (Fixes KeyError)
    print(f"\n📂 Carregando: {input_file}")
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Fuzzy matching for columns
    parque_col = next((c for c in df.columns if any(x in c for x in ['parque', 'nome', 'name'])), None)
    capacidade_col = next((c for c in df.columns if any(x in c for x in ['capacidade', 'capacity', 'cap'])), None)
    ocupacao_col = next((c for c in df.columns if any(x in c for x in ['ocupacao', 'occupancy', 'occ'])), None)
    timestamp_col = next((c for c in df.columns if any(x in c for x in ['timestamp', 'data', 'dtm_local', 'hora', 'time'])), None)
    coord_col = next((c for c in df.columns if c in ['coordinates', 'position']), None)

    # Rename for consistency
    df = df.rename(columns={parque_col: 'parque', ocupacao_col: 'ocupacao', timestamp_col: 'timestamp'})
    if capacidade_col:
        df = df.rename(columns={capacidade_col: 'capacidade'})

    # 2. Extract Coordinates
    if coord_col:
        coords = df[coord_col].apply(extract_coordinates)
        df['lat'] = coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
        df['lon'] = coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
        print(f"✓ Coordenadas extraídas de '{coord_col}'")
    else:
        print("❌ Erro: Coluna de coordenadas não encontrada!")
        return

    # 3. Time Normalization
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_hour'] = df['timestamp'].dt.floor('h')
    
    if 'capacidade' not in df.columns:
        df['capacidade'] = df.groupby('parque')['ocupacao'].transform('max')

    # Drop rows with missing essential data
    df = df.dropna(subset=['parque', 'lat', 'lon', 'timestamp_hour', 'ocupacao'])

    # 4. Traffic Score Calculation (Parallel)
    print("🚦 Calculando scores de tráfego por proximidade...")
    traffic_df = pd.read_csv(traffic_score_file)
    traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
    traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        # We use a list of tuples to pass multiple arguments to the worker
        args_list = [(row, traffic_df) for row in df.to_dict('records')]
        df['traffic_score'] = list(executor.map(traffic_score_row, args_list, chunksize=100))

    # 5. Final Hourly Aggregation
    df['data'] = df['timestamp_hour'].dt.date
    df['hora'] = df['timestamp_hour'].dt.hour
    
    # Simple mean occupancy per hour per park
    hourly = df.groupby(['parque', 'capacidade', 'data', 'hora'], as_index=False).agg({
        'ocupacao': 'mean',
        'traffic_score': 'mean',
        'lat': 'first',
        'lon': 'first'
    })

    hourly['ocupacao'] = hourly['ocupacao'].round().astype(int)
    hourly['traffic_score'] = hourly['traffic_score'].round(2)

    # 6. Save result
    hourly.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Concluído! Arquivo salvo em: {output_file}")
    print(f"📊 Registros processados: {len(hourly)}")

if __name__ == "__main__":
    preprocess_parking_data_simplified(
        input_file=r'data\parques_data\parques-estacionamento-1s-2022.xlsx',
        output_file='parking_occupancy_traffic_2022.csv',
        traffic_score_file=r'data\traffic\traffic_score_by_hour.csv',
        nthreads=16
    )