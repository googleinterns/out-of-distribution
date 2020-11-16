import random
from typing import Tuple

from PIL.Image import Image as PilImage


class RandomCrops:
    max_crop_size: int

    def __init__(self, max_crop_size: int):
        if max_crop_size <= 0:
            raise ValueError("Max_crop_size must be positive!")
        self.max_crop_size = max_crop_size

    def __call__(self, image: PilImage, targets: PilImage) -> Tuple[PilImage, PilImage]:
        crop_size = min(image.height, image.width, self.max_crop_size)

        left = random.randint(0, image.width - crop_size)
        top = random.randint(0, image.height - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        image = image.crop((left, top, right, bottom))
        targets = targets.crop((left, top, right, bottom))
        return image, targets
