from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.faster_rcnn.anchors import AnchorGenerator


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.anchor_generator = AnchorGenerator()
        self.head = RpnHead()

    def forward(self, image: torch.Tensor, features_list: List[torch.Tensor]) -> TODO:
        result = self.head(features_list)
        anchors_list = self.anchor_generator(image, features_list)

        proposals_list = [
            self.decode_deltas(reg_deltas, anchors)
            for reg_deltas, anchors in zip(result.reg_deltas_list, anchors_list)
        ]

    def decode_deltas(self, deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        pass


class RpnResult:
    obj_logits_list: List[torch.Tensor]
    reg_deltas_list: List[torch.Tensor]

    def __init__(self):
        self.obj_logits_list = []
        self.reg_deltas_list = []


class RpnHead(nn.Module):
    conv: nn.Conv2d
    classifier: nn.Conv2d
    regressor: nn.Conv2d
    out_channels: Tuple[int, int]

    def __init__(self, in_channels: int, n_anchors: int):
        super().__init__()

        self.init_layers(in_channels, n_anchors)
        self.reset_parameters()
        self.out_channels = n_anchors, n_anchors * 4

    def init_layers(self, in_channels: int, n_anchors: int) -> None:
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.classifier = nn.Conv2d(in_channels, n_anchors, 1)
        self.regressor = nn.Conv2d(in_channels, n_anchors * 4, 1)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, features_list: torch.Tensor) -> RpnResult:
        result = RpnResult()
        for x in features_list:
            x = self.conv(x)
            x = F.relu(x)
            c = self.classifier(x)
            r = self.regressor(x)

            result.obj_logits_list.append(c)
            result.reg_deltas_list.append(r)
        return result
