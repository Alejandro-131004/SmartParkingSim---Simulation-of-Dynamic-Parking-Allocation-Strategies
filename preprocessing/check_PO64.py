import pandas as pd

# Load the dataset
df = pd.read_csv("data/dataset_fluxos_hourly_2022.csv")

# Normalize
df.columns = df.columns.str.lower().str.strip()

# Filter only P064
df_064 = df[df["parking_lot"] == "P064"].copy()

# Convert date properly
df_064["date"] = pd.to_datetime(df_064["date"]).dt.date

# Expected hours (0–23)
expected_hours = set(range(24))

reports = []

print("\nChecking hourly consistency for P064...\n")

for day, group in df_064.groupby("date"):
    hours = set(group["hour"].unique())

    missing = sorted(list(expected_hours - hours))
    extra = sorted(list(hours - expected_hours))
    duplicates = group["hour"].duplicated().sum()

    reports.append({
        "date": day,
        "total_rows": len(group),
        "has_all_hours": hours == expected_hours,
        "missing_hours": ",".join(map(str, missing)) if missing else "",
        "extra_hours": ",".join(map(str, extra)) if extra else "",
        "duplicated_hours": int(duplicates)
    })

report_df = pd.DataFrame(reports)

# Save to file
report_df.to_csv("P064_hour_consistency.csv", index=False)

print("Done! Output saved to: P064_hour_consistency.csv")
print("Summary:")
print(report_df.head(10))
