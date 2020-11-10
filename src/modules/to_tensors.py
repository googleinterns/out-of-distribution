from typing import Tuple

import numpy as np
import torch
from PIL.Image import Image as PilImage
from torchvision import transforms


class ToTensors:
    image_transform: transforms.ToTensor

    def __init__(self):
        self.image_transform = transforms.ToTensor()

    def __call__(self, image: PilImage, targets: PilImage) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.image_transform(image)
        targets = self.targets_transform(targets)
        return image, targets

    def targets_transform(self, targets: PilImage) -> torch.Tensor:
        # noinspection PyTypeChecker
        array = np.asarray(targets)
        tensor = torch.from_numpy(array)
        return tensor.long()
