from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class BottleneckBlockV2S4(nn.Module):
    bn1: Optional[nn.BatchNorm2d]
    conv1: nn.Conv2d
    bn2: nn.BatchNorm2d
    conv2: nn.Conv2d
    bn3: nn.BatchNorm2d
    conv3: nn.Conv2d
    projection: Optional[nn.Conv2d]
    out_channels: int

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        if in_channels == out_channels and stride == 1:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.projection = None
        else:
            self.bn1 = None
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            skip = x
            x = self.bn1(x)
            x = F.relu(x, inplace=True)
        else:
            skip = self.projection(x)

        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)

        x = x + skip
        return x


def create_bottleneck_stage_v2s4(
    n_blocks: int, in_channels: int, mid_channels: int, out_channels: int, stride: int
) -> nn.Sequential:
    blocks = [
        BottleneckBlockV2S4(in_channels, mid_channels, out_channels, stride) if i == 0
        else BottleneckBlockV2S4(out_channels, mid_channels, out_channels, 1)
        for i in range(n_blocks)
    ] + [
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*blocks)
