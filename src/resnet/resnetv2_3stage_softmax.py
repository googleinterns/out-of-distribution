from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.normalize import Normalize
from src.resnet.bottleneck_block_v2s3 import create_bottleneck_stage_v2s3
from src.resnet.shared import ResNet_Softmax, SoftmaxMode


class ResNetV2_3Stage_Softmax(ResNet_Softmax):
    """
    Implements ResNet-v2 for classification on 32x32 RGB images (e.g. CIFAR-10).

    Reference papers:
    - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)

    Reference implementations:
    - Official code (https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua)
    - Don't use the Torch (https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/preresnet.lua)
      implementation. It uses pre-activated residual blocks instead of pre-activated bottleneck blocks, thus
      contradicting the paper which always calls for pre-activated bottleneck blocks.
    """
    normalize: Normalize
    conv1: nn.Conv2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    bn_post: nn.BatchNorm2d
    avgpool: nn.AdaptiveAvgPool2d
    fc: nn.Linear
    out_channels: int

    def __init__(self, stage_sizes: List[int], n_classes: int):
        super().__init__()

        if len(stage_sizes) != 3:
            raise ValueError("Stage_sizes must have length 3!")
        if n_classes <= 1:
            raise ValueError("N_classes must be greater than 1!")

        self.init_layers(stage_sizes, n_classes)
        self.reset_parameters()
        self.out_channels = n_classes

    def init_layers(self, stage_sizes: List[int], n_classes: int) -> None:
        self.normalize = Normalize(3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        self.stage2 = create_bottleneck_stage_v2s3(stage_sizes[0], 16, 16, 64, 1)
        self.stage3 = create_bottleneck_stage_v2s3(stage_sizes[1], 64, 32, 128, 2)
        self.stage4 = create_bottleneck_stage_v2s3(stage_sizes[2], 128, 64, 256, 2)

        self.bn_post = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, mode: SoftmaxMode) -> torch.Tensor:
        if x.shape[1:] != (3, 32, 32):
            raise ValueError("Input tensor must have shape [N, C=3, H=32, W=32]!")

        x = self.normalize(x)
        x = self.conv1(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn_post(x)
        x = F.relu(x, inplace=True)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if mode == SoftmaxMode.LOGITS:
            return x
        elif mode == SoftmaxMode.LOG_SOFTMAX:
            return F.log_softmax(x, dim=1)
        else:
            assert mode == SoftmaxMode.SOFTMAX
            return F.softmax(x, dim=1)


class ResNet29V2_Softmax(ResNetV2_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([3, 3, 3], n_classes)


class ResNet47V2_Softmax(ResNetV2_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([5, 5, 5], n_classes)


class ResNet65V2_Softmax(ResNetV2_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([7, 7, 7], n_classes)


class ResNet83V2_Softmax(ResNetV2_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([9, 9, 9], n_classes)
