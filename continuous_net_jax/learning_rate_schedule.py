from typing import List, Optional


class LearningRateSchedule:
    initial_rate: float
    decay_rate: float
    epochs: List[int]

    def __init__(self, initial_rate: float, decay_rate: float = 1.0,
                  epochs: Optional[List[int]] = None):
        if not epochs:
            epochs = []
        self.initial_rate = initial_rate
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.epochs.sort()

    def __call__(self, epoch):
        for i, decay_epoch in enumerate(self.epochs):
            if epoch < decay_epoch:
                return self.decay_rate**i * self.initial_rate
        return self.decay_rate**len(self.epochs) * self.initial_rate
