from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.normalize import Normalize
from src.resnet.bottleneck_block_v2s4 import create_bottleneck_stage_v2s4
from src.resnet.shared import ResNet_Softmax, SoftmaxMode


class ResNetV2_4Stage_Softmax(ResNet_Softmax):
    normalize: Normalize
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    stage5: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    fc: nn.Linear
    out_channels: int

    def __init__(self, stage_sizes: List[int], n_classes: int):
        super().__init__()

        if len(stage_sizes) != 4:
            raise ValueError("Stage_sizes must have length 4!")
        if n_classes <= 1:
            raise ValueError("N_classes must be greater than 1!")

        self.init_layers(stage_sizes, n_classes)
        self.reset_parameters()
        self.out_channels = n_classes

    def init_layers(self, stage_sizes: List[int], n_classes: int) -> None:
        self.normalize = Normalize(3)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage2 = create_bottleneck_stage_v2s4(stage_sizes[0], 64, 64, 256, 1)
        self.stage3 = create_bottleneck_stage_v2s4(stage_sizes[1], 256, 128, 512, 2)
        self.stage4 = create_bottleneck_stage_v2s4(stage_sizes[2], 512, 256, 1024, 2)
        self.stage5 = create_bottleneck_stage_v2s4(stage_sizes[3], 1024, 512, 2048, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_classes)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, mode: SoftmaxMode) -> torch.Tensor:
        if x.shape[1:] != (3, 224, 224):
            raise ValueError("Input tensor must have shape [N, C=3, H=224, W=224]!")

        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = F.max_pool2d(x, 3, stride=2, padding=1)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

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


class ResNet50V2_Softmax(ResNetV2_4Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([3, 4, 6, 3], n_classes)


class ResNet101V2_Softmax(ResNetV2_4Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([3, 4, 23, 3], n_classes)


class ResNet152V2_Softmax(ResNetV2_4Stage_Softmax):
    def __init__(self, n_classes: int):
        super().__init__([3, 8, 36, 3], n_classes)
