import openpyxl
import csv
import sys

INPUT_FILE = "data/parques-estacionamento-1s-2022.xlsx"
OUTPUT_FILE = "data/parques_coords.csv"

print(f"Opening {INPUT_FILE}...", flush=True)
wb = openpyxl.load_workbook(INPUT_FILE, read_only=True)
ws = wb.active

print("Converting to CSV (unique parks only)...", flush=True)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Get header
    rows = ws.rows
    header = [cell.value for cell in next(rows)]
    
    # Find indices for 'id_parque' and 'position'
    try:
        id_idx = header.index("id_parque")
        pos_idx = header.index("position")
    except ValueError:
        print(f"Error: Columns not found in {header}")
        sys.exit(1)
        
    writer.writerow(["id_parque", "position"])
    
    seen_ids = set()
    count = 0
    unique_count = 0
    
    for row in rows:
        id_val = row[id_idx].value
        if id_val in seen_ids:
            continue
            
        pos_val = row[pos_idx].value
        if pos_val: # Ensure position is not empty
            writer.writerow([id_val, pos_val])
            seen_ids.add(id_val)
            unique_count += 1
            
        count += 1
        if count % 10000 == 0:
            print(f"   Scanned {count} rows. Found {unique_count} unique parks...", flush=True)

print(f"Done! Scanned {count} rows. Saved {unique_count} unique parks to {OUTPUT_FILE}")
