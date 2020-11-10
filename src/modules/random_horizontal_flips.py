from typing import Tuple

import PIL
from PIL.Image import Image as PilImage

from src.misc.utils import random_bool


class RandomHorizontalFlips:
    flip_prob: float

    def __init__(self, flip_prob: float):
        if not (0 <= flip_prob <= 1):
            raise ValueError("Flip_prob must be in [0, 1]!")
        self.flip_prob = flip_prob

    def __call__(self, image: PilImage, targets: PilImage) -> Tuple[PilImage, PilImage]:
        if random_bool(self.flip_prob):
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            targets = targets.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return image, targets
