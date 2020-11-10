from typing import List

import torch
from torch import nn

from src.faster_rcnn.fpn import FeaturePyramidNetwork
from src.resnet.resnet_4stage_backbone import ResNet50_Backbone


class FasterRcnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNet50_Backbone()
        self.fpn = FeaturePyramidNetwork(self.resnet.out_channels, 256)

    def forward(self, image: torch.Tensor) -> None:   # TODO: return type
        features = self.resnet(image)
        features = self.fpn(features)

        proposals = self.rpn(features)
        detections = self.roi_head(features, proposals)

        return proposals, detections   # TODO: calculate loss outside of this model
