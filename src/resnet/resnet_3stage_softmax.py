from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.normalize import Normalize
from src.resnet.residual_block import create_residual_stage
from src.resnet.shared import ResNet_Softmax, SoftmaxMode


class ResNet_3Stage_Softmax(ResNet_Softmax):
    """
    Implements ResNet-v1 for classification on 32x32 RGB images (e.g. CIFAR-10).

    Reference papers:
    - Deep Residual Learning For Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)

    Reference implementations:
    - akamaster (https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)
    """
    normalize: Normalize
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
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
        self.bn1 = nn.BatchNorm2d(16)

        self.stage2 = create_residual_stage(stage_sizes[0], 16, 16, 1)
        self.stage3 = create_residual_stage(stage_sizes[1], 16, 32, 2)
        self.stage4 = create_residual_stage(stage_sizes[2], 32, 64, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_classes)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, mode: SoftmaxMode) -> torch.Tensor:
        if x.shape[1:] != (3, 32, 32):
            raise ValueError("Input tensor must have shape [N, C=3, H=32, W=32]!")

        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

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


class ResNet20_Softmax(ResNet_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([3, 3, 3], n_classes)


class ResNet32_Softmax(ResNet_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([5, 5, 5], n_classes)


class ResNet44_Softmax(ResNet_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([7, 7, 7], n_classes)


class ResNet56_Softmax(ResNet_3Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([9, 9, 9], n_classes)
