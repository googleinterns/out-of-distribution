from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class MultiResolutionFusion(nn.Module):
    conv_list = Optional[nn.ModuleList]
    out_channels: int

    def __init__(self, in_channels_list: List[int]):
        super().__init__()

        self.out_channels = min(in_channels_list)
        self.init_layers(in_channels_list)

    def init_layers(self, in_channels_list: List[int]) -> None:
        if len(in_channels_list) == 1:
            self.conv_list = None
        else:
            self.conv_list = nn.ModuleList([
                nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
                for in_channels in in_channels_list
            ])

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        if len(self.conv_list) != len(features_list):
            raise ValueError("")

        if self.conv_list is None:
            return features_list[0]   # if there is only one input feature map, it goes through this block unchanged

        max_width = max(features.size(2) for features in features_list)
        max_height = max(features.size(3) for features in features_list)
        out_size = max_width, max_height

        features_list = [conv(feature) for conv, feature in zip(self.conv_list, features_list)]
        features_list = [F.interpolate(features, size=out_size, mode="bilinear") for features in features_list]
        return torch.stack(features_list, dim=0).sum(dim=0)
