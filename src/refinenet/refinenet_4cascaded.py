import torch
from torch import nn
from torch.nn import functional as F

from src.refinenet.refine_block import RefineBlock
from src.resnet.resnet_4stage_backbone import ResNet101_Backbone


class RefineNet_4Cascaded(nn.Module):
    """
    Implements RefineNet 4-cascaded for semantic segmentation on RGB images.

    Reference papers:
    - RefineNet: Multi-Path Refinement Networks For High-Resolution Semantic Segmentation
      (https://arxiv.org/pdf/1611.06612.pdf)
    - RefineNet: Multi-Path Refinement Networks For Dense Prediction (https://ieeexplore.ieee.org/document/8618363)
    """
    backbone: ResNet101_Backbone
    block1: RefineBlock
    block2: RefineBlock
    block3: RefineBlock
    block4: RefineBlock
    out_channels: int

    def __init__(self, n_classes: int):
        super().__init__()
        self.init_layers()
        self.out_channels = n_classes

    def init_layers(self) -> None:
        self.backbone = ResNet101_Backbone()

        self.block4 = RefineBlock([self.backbone.out_channels[3]], False)
        self.block3 = RefineBlock([self.backbone.out_channels[2], self.block4.out_channels], False)
        self.block2 = RefineBlock([self.backbone.out_channels[1], self.block3.out_channels], False)
        self.block1 = RefineBlock([self.backbone.out_channels[0], self.block2.out_channels], True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)

        x = self.block4([features[3]])
        x = self.block3([features[2], x])
        x = self.block2([features[1], x])
        x = self.block1([features[0], x])

        x = F.interpolate(x, size=images.shape[2:], mode="bilinear")
        x = F.log_softmax(x, dim=1)
        return x
