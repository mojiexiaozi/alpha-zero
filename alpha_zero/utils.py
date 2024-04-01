from dataclasses import dataclass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:.2e}"


@dataclass
class Config:
    num_iters: int = 1000
    num_episodes: int = 100
    temp_threshold: int = 15
    update_threshold: int = 0.6
    maxlen_of_queue: int = 200000
    num_mcts_sims: int = 25
    arena_compare: int = 40
    cpuct: int = 1

    checkpoint: str = "./checkpoint"
    load_model: bool = False
    load_folder_file: str = "./checkpoint"
    num_iters_for_training: int = 20
