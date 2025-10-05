class FCFS:
    def choose(self, lots, now):
        # escolhe o parque mais barato; empate resolve pela menor distância
        return sorted(lots, key=lambda L:(L.price, L.distance))[0] if lots else None

def make_policy(name):
    if name == "fcfs": return FCFS()
    raise ValueError(f"Unknown policy: {name}")
