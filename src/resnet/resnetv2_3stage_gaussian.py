from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.max_mahalanobis import MaxMahalanobis, GaussianResult
from src.modules.normalize import Normalize
from src.resnet.bottleneck_block_v2s3 import create_bottleneck_stage_v2s3
from src.resnet.shared import GaussianMode, ResNet_Gaussian


class ResNetV2_3Stage_Gaussian(ResNet_Gaussian):
    """
    Implements Max-Mahalanobis center loss for classification on 32x32 RGB images (e.g. CIFAR-10).

    Reference papers:
    - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
    - Rethinking Softmax Cross-Entropy Loss For Adversarial Robustness (https://arxiv.org/pdf/1905.10626.pdf)

    Reference implementations:
    - Official code (https://github.com/P2333/Max-Mahalanobis-Training/blob/master/train.py)
    """
    normalize: Normalize
    conv1: nn.Conv2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    bn_post: nn.BatchNorm2d
    avgpool: nn.AdaptiveAvgPool2d
    fc: nn.Linear
    max_mahalanobis: MaxMahalanobis
    out_channels: int

    def __init__(self, stage_sizes: List[int], radius: float, n_classes: int):
        super().__init__()

        if len(stage_sizes) != 3:
            raise ValueError("Stage_sizes must have length 3!")
        if radius <= 0:
            raise ValueError("Radius must be positive!")
        if n_classes <= 1:
            raise ValueError("N_classes must be greater than 1!")

        self.init_layers(stage_sizes, radius, n_classes)
        self.reset_parameters()
        self.out_channels = n_classes

    def init_layers(self, stage_sizes: List[int], radius: float, n_classes: int) -> None:
        self.normalize = Normalize(3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)

        self.stage2 = create_bottleneck_stage_v2s3(stage_sizes[0], 16, 16, 64, 1)
        self.stage3 = create_bottleneck_stage_v2s3(stage_sizes[1], 64, 32, 128, 2)
        self.stage4 = create_bottleneck_stage_v2s3(stage_sizes[2], 128, 64, 256, 2)

        self.bn_post = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)
        self.max_mahalanobis = MaxMahalanobis(radius, 256, n_classes)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, mode: GaussianMode) -> Union[torch.Tensor, GaussianResult]:
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

        x = self.max_mahalanobis(x, mode)
        return x


class ResNet29V2_Gaussian(ResNetV2_3Stage_Gaussian):
    def __init__(self, radius: float, n_classes: int):
        super().__init__([3, 3, 3], radius, n_classes)


class ResNet47V2_Gaussian(ResNetV2_3Stage_Gaussian):
    def __init__(self, radius: float, n_classes: int):
        super().__init__([5, 5, 5], radius, n_classes)


class ResNet65V2_Gaussian(ResNetV2_3Stage_Gaussian):
    def __init__(self, radius: float, n_classes: int):
        super().__init__([7, 7, 7], radius, n_classes)


class ResNet83V2_Gaussian(ResNetV2_3Stage_Gaussian):
    def __init__(self, radius: float, n_classes: int):
        super().__init__([9, 9, 9], radius, n_classes)
