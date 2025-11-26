import pandas as pd


class StateManager:
    """
    Loads the most recent real-world state of the dataset and returns
    as allowed exogenous and endogenous variables.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_current_state(self):
        df = pd.read_csv(self.dataset_path)
        df.columns = df.columns.str.lower().str.strip()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "hour"])

        last = df.iloc[-1]   # last line = most recent state

        return {
            "current_date": last["date"],
            "current_hour": int(last["hour"]),
            "current_occupancy": int(last["occupancy"]),
            "entries": int(last.get("entries", 0)),
            "exits": int(last.get("exits", 0)),
            "traffic_score": float(last.get("traffic_score", 0)),
            "temp_min": float(last.get("temp_min_c", 0)),
            "temp_max": float(last.get("temp_max_c", 0)),
        }