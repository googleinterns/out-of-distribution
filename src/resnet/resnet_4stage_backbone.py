from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.normalize import Normalize
from src.resnet.bottleneck_block import create_bottleneck_stage, BottleneckBlock
from src.resnet.residual_block import ResidualBlock, create_residual_stage


class ResNet_4Stage_Backbone(nn.Module):
    """
    Implements ResNet-v1 for feature extraction on 224x224 RGB images (e.g. ImageNet).

    Reference papers:
    - Deep Residual Learning For Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)

    Reference implementations:
    - PyTorch (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
    """
    normalize: Normalize
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    stage5: nn.Sequential
    out_channels: List[int]

    def __init__(self, stage_sizes: List[int], block_type: type):
        super().__init__()

        if len(stage_sizes) != 4:
            raise ValueError("Stage_sizes must have length 4!")
        if block_type != ResidualBlock and block_type != BottleneckBlock:
            raise ValueError("Block_type must be ResidualBlock or BottleneckBlock!")

        self.init_layers(stage_sizes, block_type)
        self.reset_parameters()
        self.out_channels = [stage[-1].out_channels for stage in (self.stage2, self.stage3, self.stage4, self.stage5)]

    def init_layers(self, stage_sizes: List[int], block_type: type) -> None:
        self.normalize = Normalize(3)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if block_type == ResidualBlock:
            self.stage2 = create_residual_stage(stage_sizes[0], 64, 64, 1)
            self.stage3 = create_residual_stage(stage_sizes[1], 64, 128, 2)
            self.stage4 = create_residual_stage(stage_sizes[2], 128, 256, 2)
            self.stage5 = create_residual_stage(stage_sizes[3], 256, 512, 2)
        else:
            self.stage2 = create_bottleneck_stage(stage_sizes[0], 64, 64, 256, 1)
            self.stage3 = create_bottleneck_stage(stage_sizes[1], 256, 128, 512, 2)
            self.stage4 = create_bottleneck_stage(stage_sizes[2], 512, 256, 1024, 2)
            self.stage5 = create_bottleneck_stage(stage_sizes[3], 1024, 512, 2048, 2)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.size(2) % 32 != 0:
            raise ValueError("Input tensor's height must be divisible by 32!")
        if x.size(3) % 32 != 0:
            raise ValueError("Input tensor's width must be divisible by 32!")

        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = F.max_pool2d(x, 3, stride=2, padding=1)

        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return [x2, x3, x4, x5]


class ResNet18_Backbone(ResNet_4Stage_Backbone):
    def __init__(self):
        super().__init__([2, 2, 2, 2], ResidualBlock)


class ResNet34_Backbone(ResNet_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 4, 6, 3], ResidualBlock)


class ResNet50_Backbone(ResNet_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 4, 6, 3], BottleneckBlock)


class ResNet101_Backbone(ResNet_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 4, 23, 3], BottleneckBlock)


class ResNet152_Backbone(ResNet_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 8, 36, 3], BottleneckBlock)
