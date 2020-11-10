from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.misc.utils import reversed_zip


class FeaturePyramidNetwork(nn.Module):
    lateral_conv: nn.ModuleList
    final_conv: nn.ModuleList
    out_channels: int

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()

        self.init_layers(in_channels_list, out_channels)
        self.reset_parameters()
        self.out_channels = out_channels

    def init_layers(self, in_channels_list: List[int], out_channels: int) -> None:
        self.lateral_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
        ])
        self.final_conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)   # leaky ReLU with a=1 is the identity function
                nn.init.constant_(module.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []   # accumulate top-down feature maps from highest to lowest semantic content
        top_down_fmap = None

        for bottom_up_fmap, lateral_conv, final_conv in reversed_zip(x, self.lateral_conv, self.final_conv):
            lateral_fmap = lateral_conv(bottom_up_fmap)

            if top_down_fmap is None:
                top_down_fmap = lateral_fmap
            else:
                top_down_fmap = F.interpolate(top_down_fmap, scale_factor=2, mode="nearest")
                top_down_fmap = top_down_fmap + lateral_fmap

            final_fmap = final_conv(top_down_fmap)
            result.append(final_fmap)

        result.reverse()   # reorder top-down feature maps from lowest to highest semantic content

        last_level_fmap = F.max_pool2d(result[-1], 1, 2, 0)
        result.append(last_level_fmap)
        return result
