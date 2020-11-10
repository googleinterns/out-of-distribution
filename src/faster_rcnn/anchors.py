from math import sqrt
from typing import List

import torch
from torch import nn


class AnchorGenerator(nn.Module):
    sizes: List[int]
    aspect_ratios: List[float]

    def __init__(self, sizes: List[int], aspect_ratios: List[float]):
        super().__init__()

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def forward(self, image: torch.Tensor, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(self.sizes) != len(features_list):
            raise ValueError("Sizes and features_list must have the same length!")

        result = []
        for size, features in zip(self.sizes, features_list):
            stride = image.size(2) // features.size(2)
            if stride != image.size(2) / features.size(2):
                raise ValueError("Stride must be an integer!")
            if stride != image.size(3) / features.size(3):
                raise ValueError("Horizontal and vertical stride must be the same!")

            anchors = self.generate_scale_anchors(size, stride, features)
            result.append(anchors)

        return result

    def generate_scale_anchors(self, size: int, stride: int, features: torch.Tensor) -> torch.Tensor:
        cell_anchors = self.generate_cell_anchors(size)   # [R, 4]
        centers = self.generate_centers(stride, features)   # [HW, 2]

        offsets = centers[:, [0, 0, 1, 1]]   # [HW, 4]
        scale_anchors = offsets[:, None, :] + cell_anchors[None, :, :]   # [HW, R, 4]
        scale_anchors = scale_anchors.reshape(-1, 4)   # [HWR, 4]

        return torch.round(scale_anchors).int()

    def generate_cell_anchors(self, size: int) -> torch.Tensor:
        result = []
        for aspect_ratio in self.aspect_ratios:
            height = size * sqrt(aspect_ratio)
            width = size / sqrt(aspect_ratio)
            anchor = -width / 2, width / 2, -height / 2, height / 2
            result.append(anchor)
        return torch.tensor(result)   # [R, 4]

    def generate_centers(self, stride: int, features: torch.Tensor) -> torch.Tensor:
        y = stride * torch.arange(features.size(2))   # [H]
        x = stride * torch.arange(features.size(3))   # [W]

        # noinspection PyTypeChecker
        y, x = torch.meshgrid(y, x)   # [H, W], [H, W]
        y = y.reshape(-1)   # [HW]
        x = x.reshape(-1)   # [HW]

        return torch.stack([x, y], dim=1)   # [HW, 2]
