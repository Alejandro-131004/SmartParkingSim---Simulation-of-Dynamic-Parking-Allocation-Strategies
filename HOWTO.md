# Digital Twin for Dynamic Parking Pricing

This project implements a Digital Twin framework to evaluate and compare static vs. dynamic pricing policies (Classic PID vs. AI/MCTS) for urban parking management. It uses a macroscopic simulation engine calibrated with real-world data from Lisbon (EMEL).

## 1. Project Structure
- `runner.py`: Main entry point to run simulations and compare policies.
- `digital_twin_sim/`: Core simulation logic (`twin_offline_sim.py`, `occupancy_forecast.py`, `chronos_adapter.py`).
- `mcts_agent.py`: The AI agent implementing Monte Carlo Tree Search.
- `policies.py`: Definitions of pricing strategies (Static, Dynamic, MCTS).
- `config/`: Configuration files (YAML).
- `data/`: Dataset storage.
- `reports/`: Output folder for plots and CSV results.

## 2. Prerequisites & Installation

### System Requirements
* Python 3.8 or higher.
* **Virtual Environment:** Recommended (e.g., `env_simpy`).
* **Core Libraries:** `numpy`, `pandas`, `matplotlib`, `simpy`, `PyYAML`.
* **Analysis & Utilities:** `scikit-learn`, `tqdm`.

### Installation

1. Open a terminal in the project root directory.

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv env_simpy
   source env_simpy/bin/activate  # On Windows: env_simpy\Scripts\activate
    ```
3. Install the required dependencies:
    ```
    pip install pandas==2.3.3 numpy==2.0.2 matplotlib==3.9.4 simpy==4.1.1 PyYAML==6.0.3 scikit-learn tqdm
    ```

(Standard sub-dependencies like contourpy, cycler, and pillow will be installed automatically).

**Optional (Amazon Chronos)**: To enable the Deep Learning Forecaster, you must additionally install torch and gluonts. If these are missing, the system gracefully defaults to the Statistical Forecaster.

## 3. Data Setup
The simulation relies on processed historical data. Ensure the following file exists in your data directory:

```
data/P023_sim/dataset_P023_simready_updated.csv
```

If you only have the raw Excel/CSV files, you can regenerate the simulation-ready dataset using the preprocessing pipeline:

```
# 1. Convert raw data to hourly flows
python preprocess_dataset.py

# 2. Extract P023 specific data and merge with traffic scores
python create_dataset_P023_simready.py
```

## 4. How to Run the Simulation
The `runner.py` script is the primary tool for executing experiments. It supports three modes.

### Mode 1: Classic Analysis (Calibration)
Runs the Static Pricing vs. Reactive Controller (PID).

```bash
python runner.py --mode 1
```

### Mode 2: AI Analysis
Runs the Static Pricing vs. MCTS Agent.

```bash
python runner.py --mode 2
```

### Mode 3: Battle Mode (Paper Results)
Compares Classic Control vs. AI Agent under high-demand scenarios (3× Demand Multiplier).

```bash
python runner.py --mode 3
```

### Expected Output:
- Grand Battle Report table printed in the terminal.
- Plots saved to `reports/`.
- CSV data saved to `reports/`.

## 5. Advanced Configuration (Optional)
You can override simulation parameters via the Command Line Interface (CLI):

- Target Occupancy: `--target 0.85`
- Controller Gain: `--k 0.5`
- Minimum Price: `--p_min 1.0`

Example:

```bash
python runner.py --mode 3 --target 0.85 --k 0.5
```

Alternatively, edit the permanent configuration in:

```
config/parking_P023.yaml
```

## 6. Reproducing Paper Plots
Ensure you have run the simulation at least once (Mode 3 recommended).

Run:

```bash
python analysis_deep_dive.py
```

This script generates:
- `deep_dive_1_forecast_accuracy.png`
- `deep_dive_2_price_strategy.png`
- `deep_dive_3_occupancy_result.png`
