import torch
from torch import nn
from torch.nn import functional as F


class ResidualConvUnit(nn.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    out_channels: int

    def __init__(self, in_channels: int):
        super().__init__()

        self.init_layers(in_channels)
        self.out_channels = in_channels

    def init_layers(self, in_channels: int) -> None:
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = x + skip
        return x
