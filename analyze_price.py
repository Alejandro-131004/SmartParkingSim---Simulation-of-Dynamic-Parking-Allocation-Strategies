import pandas as pd

try:
    df = pd.read_csv('reports/comparison_data.csv')
    print("Dynamic Price Min:", df['dynamic_price'].min())
    print("Dynamic Price Max:", df['dynamic_price'].max())
    print("Dynamic Price Mean:", df['dynamic_price'].mean())
    
    print("Static Price Min:", df['static_price'].min())
    print("Static Price Max:", df['static_price'].max())
except Exception as e:
    print(e)
