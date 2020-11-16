from typing import List

import torch
from torch import nn

from src.refinenet.residual_conv_unit import ResidualConvUnit


class AdaptiveConv(nn.Module):
    conv_list: nn.ModuleList
    rcus_list: nn.ModuleList
    out_channels: List[int]

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()

        self.init_layers(in_channels_list, out_channels)
        self.out_channels = [out_channels] * len(in_channels_list)

    def init_layers(self, in_channels_list: List[int], out_channels: int) -> None:
        """
        Constructs the layers of this Adaptive Conv. A dimension-adapting conv layer (self.conv_list) is placed between
        each input and its respective RCUs. This conv layer is not explicitly mentioned in the papers, but is in the
        architecture diagram (https://github.com/guosheng/refinenet/blob/master/net_graphs/part2_cascaed_refinenet.pdf).
        :param in_channels_list: the respective number of channels in each input tensor
        :param out_channels: the number of channels in the output tensors
        :return: None
        """
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
            for in_channels in in_channels_list
        ])

        self.rcus_list = nn.ModuleList([
            nn.Sequential(
                ResidualConvUnit(out_channels),
                ResidualConvUnit(out_channels)
            )
            for _ in in_channels_list
        ])

    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [rcus(features) for rcus, features in zip(self.rcus_list, features_list)]
