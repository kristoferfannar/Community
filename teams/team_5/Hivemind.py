class Hivemind:
    def __init__(self, phase1_optimal_pair: list[tuple[tuple, int]] = None, phase2_optimal_pair: list[tuple] = None, prev_task:int = -1):
        self.phase1_optimal_pair = phase1_optimal_pair
        self.phase2_optimal_pair = phase2_optimal_pair
        self.prev_task = prev_task