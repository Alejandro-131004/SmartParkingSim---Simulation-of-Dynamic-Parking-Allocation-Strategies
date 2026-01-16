import pandas as pd

df_global = pd.read_csv("data/dataset_fluxos_hourly_2022_updated.csv")
df_indiv = pd.read_csv("data/dataset_P023_simready.csv")

merge_keys = ["parking_lot", "date", "hour"]

df_final = df_indiv.merge(
    df_global[merge_keys + ["traffic_score"]],
    on=merge_keys,
    how="left",
    suffixes=("", "_from_global"),
)

# Always overwrite and round
df_final["traffic_score"] = df_final["traffic_score_from_global"].round(1)
df_final.drop(columns=["traffic_score_from_global"], inplace=True)

df_final.to_csv("dataset_P023_simready_updated.csv", index=False)

print("Column 'traffic_score' copied and rounded to 1 decimal place.")
