from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.normalize import Normalize
from src.resnet.bottleneck_block_v2s4 import create_bottleneck_stage_v2s4


class ResNetV2_4Stage_Backbone(nn.Module):
    """
    Implements ResNet-v2 for feature extraction on 224x224 RGB images (e.g. ImageNet).

    Reference papers:
    - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)

    Reference implementations:
    - Torch (https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/preresnet.lua). This implementation
      is endorsed by Kaiming He at (https://github.com/KaimingHe/resnet-1k-layers/issues/2#issuecomment-235209300).
    - Don't use the TensorFlow (https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) or
      Keras (https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)
      implementations. They place stride-2 convolutions at the end of stages 2-4 rather than at the beginning of stages
      3-5 (https://github.com/tensorflow/models/blob/87e4768eaa6c0b1e77c00c5f41f8e689e566ee03/research/slim/nets/resnet_
      utils.py#L30), causing the height/width of the stage outputs to differ from ResNet-v1. Unfortunately this means we
      cannot utilize Keras's pretrained weights.
    """
    normalize: Normalize
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    stage5: nn.Sequential
    out_channels: List[int]

    def __init__(self, stage_sizes: List[int]):
        super().__init__()

        if len(stage_sizes) != 4:
            raise ValueError("Stage_sizes must have length 4!")

        self.init_layers(stage_sizes)
        self.reset_parameters()
        self.out_channels = [stage[-1].out_channels for stage in (self.stage2, self.stage3, self.stage4, self.stage5)]

    def init_layers(self, stage_sizes: List[int]) -> None:
        self.normalize = Normalize(3)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage2 = create_bottleneck_stage_v2s4(stage_sizes[0], 64, 64, 256, 1)
        self.stage3 = create_bottleneck_stage_v2s4(stage_sizes[1], 256, 128, 512, 2)
        self.stage4 = create_bottleneck_stage_v2s4(stage_sizes[2], 512, 256, 1024, 2)
        self.stage5 = create_bottleneck_stage_v2s4(stage_sizes[3], 1024, 512, 2048, 2)

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


class ResNet50V2_Backbone(ResNetV2_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 4, 6, 3])


class ResNet101V2_Backbone(ResNetV2_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 4, 23, 3])


class ResNet152V2_Backbone(ResNetV2_4Stage_Backbone):
    def __init__(self):
        super().__init__([3, 8, 36, 3])
