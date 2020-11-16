from typing import List

import torch


class SummaryStatistics:
    min: float
    mean: float
    median: float
    max: float

    def __init__(self, elements: List[float]):
        elements = torch.tensor(elements, dtype=torch.float32)
        self.min = elements.min().item()
        self.mean = elements.mean().item()
        self.median = elements.median().item()
        self.max = elements.max().item()
