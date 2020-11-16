from typing import List

import torch
from torch import nn


class Normalize(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, n_features: int):
        super().__init__()

        # initialize to NaN so the code fails hard when this layer has not been initialized
        mean = torch.full([n_features], float("nan"))
        std = torch.full([n_features], float("nan"))

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def set_parameters(self, mean: List[float], std: List[float]) -> None:
        if self.mean.size(0) != len(mean):
            raise ValueError("Mean has wrong number of features!")
        if self.std.size(0) != len(std):
            raise ValueError("Std has wrong number of features!")

        for j, channel_mean in enumerate(mean):
            self.mean[j] = channel_mean
        for j, channel_std in enumerate(std):
            self.std[j] = channel_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean[:, None, None]) / self.std[:, None, None]
