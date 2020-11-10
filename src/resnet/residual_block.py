from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    projection: Optional[nn.Sequential]
    out_channels: int

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.projection = None
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.projection is None else self.projection(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x += skip
        x = F.relu(x, inplace=True)
        return x


def create_residual_stage(n_blocks: int, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    blocks = [
        ResidualBlock(in_channels, out_channels, stride) if i == 0
        else ResidualBlock(out_channels, out_channels, 1)
        for i in range(n_blocks)
    ]
    return nn.Sequential(*blocks)
