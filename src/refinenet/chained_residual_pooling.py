import torch
from torch import nn
from torch.nn import functional as F


class ChainedResidualPooling(nn.Module):
    pooling_block_list: nn.ModuleList
    out_channels: int

    def __init__(self, n_pooling_blocks: int, n_channels: int):
        super().__init__()

        self.init_layers(n_pooling_blocks, n_channels)
        self.out_channels = n_channels

    def init_layers(self, n_pooling_blocks: int, n_channels: int) -> None:
        self.pooling_block_list = nn.ModuleList([
            create_pooling_block(n_channels) for _ in range(n_pooling_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        result = x

        for pooling_block in self.pooling_block_list:
            x = pooling_block(x)
            result = result + x

        return result


def create_pooling_block(n_channels: int) -> nn.Sequential:
    """
    Constructs a pooling block for use in Chained Residual Pooling. We use the improved architecture described in the
    IEEE paper, with the conv layer before the max-pool layer.
    :param n_channels: the number of channels in the input and output tensors
    :return: the constructed pooling block
    """
    return nn.Sequential(
        nn.Conv2d(n_channels, n_channels, 3, padding=1),
        nn.MaxPool2d(5, stride=1, padding=2)
    )
