from typing import List

import torch
from torch import nn

from src.refinenet.adaptive_conv import AdaptiveConv
from src.refinenet.chained_residual_pooling import ChainedResidualPooling
from src.refinenet.multi_resolution_fusion import MultiResolutionFusion
from src.refinenet.residual_conv_unit import ResidualConvUnit


class RefineBlock(nn.Module):
    adaptive_conv: AdaptiveConv
    multi_resolution_fusion: MultiResolutionFusion
    chained_residual_pooling: ChainedResidualPooling
    output_conv: nn.Sequential
    out_channels: int

    def __init__(self, in_channels_list: List[int], block_index: int):
        super().__init__()

        self.init_layers(in_channels_list, block_index)
        self.reset_parameters()
        self.out_channels = self.output_conv.out_channels

    def init_layers(self, in_channels_list: List[int], block_index: int) -> None:
        """
        Constructs the layers of this RefineBlock. We use 4 pooling blocks in Chained Residual Pooling, as per the IEEE
        paper.
        :param in_channels_list: the respective number of channels in each input tensor
        :param block_index: the index of this RefineBlock, according to the convention in the paper (RefineBlock-1 takes
        in the highest resolution features; RefineBlock-4 takes in the lowest resolution features)
        :return: None
        """
        self.adaptive_conv = AdaptiveConv(in_channels_list, 512 if block_index == 4 else 256)
        self.multi_resolution_fusion = MultiResolutionFusion(self.adaptive_conv.out_channels)
        self.chained_residual_pooling = ChainedResidualPooling(4, self.multi_resolution_fusion.out_channels)
        self.output_conv = create_output_conv(self.chained_residual_pooling.out_channels, 3 if block_index == 1 else 1)
    
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.normal_(module.bias, std=0.01)

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        features_list = self.adaptive_conv(features_list)
        features = self.multi_resolution_fusion(features_list)
        features = self.chained_residual_pooling(features)
        features = self.output_conv(features)
        return features


def create_output_conv(n_channels: int, n_rcus: int) -> nn.Sequential:
    blocks = [ResidualConvUnit(n_channels) for _ in range(n_rcus)]
    return nn.Sequential(*blocks)
