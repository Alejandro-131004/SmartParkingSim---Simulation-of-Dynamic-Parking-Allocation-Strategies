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
    traffic_hour = traffic_df[
        (traffic_df['date'] == pd.to_datetime(ts).date()) & (traffic_df['hour'] == pd.to_datetime(ts).hour)
    ]
    if traffic_hour.empty:
        return np.nan
    rads = np.radians
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
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
        weights = 1 / (traffic_near['distance_km'].replace(0,0.1)**2)
        score = np.average(traffic_near['score'], weights=weights)
        return round(float(score),2)
    return np.nan

# Move this OUTSIDE the preprocess function!
def traffic_score_row(args):
    row, traffic_df = args
    return get_weighted_traffic_score(row['timestamp_hour'], row['lat'], row['lon'], traffic_df)

def preprocess_parking_data_xlsx(
        input_file, output_file, traffic_score_file,
        flux_multiplier=1.10, entry_exit_ratio=0.30, min_flux_ratio=0.10, nthreads=8):

    print("\n" + "="*70)
    print(" PRE-PROCESSAMENTO INTEGRADO COM SCORE DE TRÁFEGO")
    print("="*70)
    print(f"\n📂 Carregando estacionamento...")
    df = pd.read_excel(input_file, sheet_name=0)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    parque_col = next((c for c in df.columns if any(x in c for x in ['parque', 'nome', 'name'])), None)
    capacidade_col = next((c for c in df.columns if any(x in c for x in ['capacidade', 'capacity', 'cap'])), None)
    ocupacao_col = next((c for c in df.columns if any(x in c for x in ['ocupacao', 'occupancy', 'occ'])), None)
    timestamp_col = next((c for c in df.columns if any(x in c for x in ['timestamp', 'data', 'dtm_local', 'hora', 'time'])), None)
    coord_col = next((c for c in df.columns if c in ['coordinates', 'position']), None)
    df = df.rename(columns={parque_col: 'parque', ocupacao_col: 'ocupacao', timestamp_col: 'timestamp'})
    if capacidade_col:
        df = df.rename(columns={capacidade_col: 'capacidade'})
    if coord_col:
        coords = df[coord_col].apply(extract_coordinates)
        df['lat'] = coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
        df['lon'] = coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
        print(f"✓ Coordenadas extraídas de {coord_col}")
    else:
        print("❌ Nenhuma coluna de coordenada encontrada!")
        sys.exit(1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_hour'] = df['timestamp'].dt.floor('h')
    if 'capacidade' not in df.columns:
        df['capacidade'] = df.groupby('parque')['ocupacao'].transform('max')
    df = df.dropna(subset=['parque','lat','lon','timestamp_hour','ocupacao'])

    traffic_df = pd.read_csv(traffic_score_file)
    traffic_df['date'] = pd.to_datetime(traffic_df['date']).dt.date
    traffic_df['hour'] = pd.to_numeric(traffic_df['hour'], errors='coerce')
    print("\n🚦 Calculando score de tráfego ponderado por proximidade por hora...")

    # Pass both the row and traffic_df as tuple to the worker function
    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        df['traffic_score'] = list(executor.map(traffic_score_row, [(row, traffic_df) for row in df.to_dict('records')], chunksize=100))

    df['ocupacao'] = df['ocupacao'].fillna(0).astype(int)
    df['ocupacao_anterior'] = df.groupby('parque')['ocupacao'].shift(1).fillna(0).astype(int)
    df['variacao_liquida'] = df['ocupacao'] - df['ocupacao_anterior']
    df['fluxo_base'] = df.apply(lambda row: abs(row['variacao_liquida']) if row['variacao_liquida'] != 0 else max(1, int(row['ocupacao'] * min_flux_ratio)), axis=1)
    df['fluxo_oculto'] = (df['fluxo_base'] * (flux_multiplier - 1.0)).round().astype(int)
    df['entradas_ocultas'] = (df['fluxo_oculto'] * entry_exit_ratio).round().astype(int)
    df['saidas_ocultas'] = (df['fluxo_oculto'] * (1 - entry_exit_ratio)).round().astype(int)
    def calc_fluxos(row):
        var = row['variacao_liquida']
        if var > 0:
            return pd.Series({'entradas': int(var + row['saidas_ocultas']), 'saidas': int(row['saidas_ocultas'])})
        elif var < 0:
            return pd.Series({'entradas': int(row['entradas_ocultas']), 'saidas': int(abs(var) + row['entradas_ocultas'])})
        else:
            return pd.Series({'entradas': int(row['entradas_ocultas']), 'saidas': int(row['saidas_ocultas'])})
    fluxos = df.apply(calc_fluxos, axis=1)
    df['entradas'] = fluxos['entradas']
    df['saidas'] = fluxos['saidas']

    df['data'] = df['timestamp_hour'].dt.date
    df['hora'] = df['timestamp_hour'].dt.hour
    df['dia'] = df['timestamp_hour'].dt.dayofyear
    df['dia_semana'] = df['timestamp_hour'].dt.dayofweek

    hourly = df.groupby(['parque','capacidade','dia','data','hora','dia_semana'], as_index=False).agg({
        'entradas':'sum',
        'saidas':'sum',
        'ocupacao':'mean',
        'traffic_score':'mean',
    })
    hourly['ocupacao'] = hourly['ocupacao'].round().astype(int)
    hourly['entradas'] = hourly['entradas'].astype(int)
    hourly['saidas'] = hourly['saidas'].astype(int)
    hourly['fluxo_total'] = hourly['entradas'] + hourly['saidas']
    hourly['taxa_fluxo_hora'] = hourly['fluxo_total'].astype(float)

    hourly.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Salvo: {output_file}")
    print(f"\n📈 Resumo: {len(hourly)} registos. Parques: {hourly['parque'].nunique()}")
    print(f"   Período: {hourly['data'].min()} a {hourly['data'].max()}")
    print(hourly[['parque','data','hora','traffic_score']].sample(5))

if __name__ == "__main__":
    preprocess_parking_data_xlsx(
        input_file='data/parques-estacionamento-1s-2022.xlsx',
        output_file='dataset_fluxos_hourly_2022.csv',
        traffic_score_file='data/traffic_score_by_hour.csv',
        flux_multiplier=1.10,
        entry_exit_ratio=0.30,
        min_flux_ratio=0.10,
        nthreads=8
    )
