import os
import yaml
import shutil
import itertools
import subprocess
import pandas as pd
import json
import sys


# Resolve path correctly no matter where script is run
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_YAML = os.path.join(ROOT, "config", "parking_P023.yaml")
TEMP_YAML = os.path.join(ROOT, "config", "tmp_scenario.yaml")
RUNNER = os.path.join(ROOT, "src", "runner.py")

# Scenarios to test
TARGETS = [0.75, 0.85, 0.95]
KS = [0.2, 0.4, 0.6]
PMINS = [1.3, 1.5, 1.7]

RESULTS = []

def run_scenario    (target, k, p_min):
    """
    Runs runner.py using the SAME Python interpreter as this script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    runner_path = os.path.join(current_dir, "runner.py")

    cmd = [
        sys.executable,  
        runner_path, 
        "--mode", "2",
        "--target", str(target),
        "--k", str(k),
        "--p_min", str(p_min),
        "--silent" 
    ]
    
    print(f"=== Running scenario: target={target}, k={k}, p_min={p_min} ===")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("!!! ERROR IN SIMULATION !!!")
            print("STDERR:", result.stderr)
            return None

        output_lines = result.stdout.strip().split('\n')
        json_line = None
        for line in reversed(output_lines):
            if line.strip().startswith('{') and "static_revenue" in line:
                json_line = line
                break
        
        if json_line:
            return json.loads(json_line)
        else:
            print(f"[Warn] No JSON output found for params: {target}, {k}, {p_min}")
            return None

    except Exception as e:
        print(f"[Error] Failed to run simulation: {e}")
        return None

def main():
    # Ensure RESULTS is treated as a list locally or global if defined outside
    # Ideally, define it here to be safe
    results_data = []

    for target, k, p_min in itertools.product(TARGETS, KS, PMINS):
        # 1. Capture the return value from the single run
        data = run_scenario(target, k, p_min)
        
        # 2. Check if data is valid (not None) before appending
        if data is not None:
            # Inject parameters into the result dict for clearer CSV analysis
            data['target'] = target
            data['k'] = k
            data['p_min'] = p_min
            
            results_data.append(data)
        else:
            print(f"[Warn] Simulation failed or returned None for: target={target}, k={k}, p_min={p_min}")

    # 3. Check if we have results before creating DataFrame
    if not results_data:
        print("[Critical] No results were captured. The RESULTS list is empty.")
        return

    df = pd.DataFrame(results_data)
    df.to_csv("scenario_results.csv", index=False)

    print("\n=== SCENARIOS COMPLETED ===")
    
    # Check if the column exists to avoid KeyError
    if "dynamic_revenue" in df.columns:
        print(df.sort_values("dynamic_revenue", ascending=False).head(10))
    else:
        print("[Error] 'dynamic_revenue' column not found. Available columns:", df.columns)
        print(df.head())


if __name__ == "__main__":
    main()
