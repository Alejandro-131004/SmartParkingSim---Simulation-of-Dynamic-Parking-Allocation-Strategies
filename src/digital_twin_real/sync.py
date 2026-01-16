class TwinSynchronizer:
    """
    Updates the SimPy simulation state based on the real-world state.
    """

    def sync_model_to_reality(self, env, lots, real_state):
        env.now = real_state["current_hour"] * 60

        for lot in lots:
            lot.res.count = real_state["current_occupancy"]
            lot.last = env.now
