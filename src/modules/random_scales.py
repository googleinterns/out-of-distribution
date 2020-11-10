from typing import Tuple

import PIL
from PIL.Image import Image as PilImage

from src.misc.utils import random_float


class RandomScales:
    lo: float
    hi: float

    def __init__(self, lo: float, hi: float):
        if lo <= 0:
            raise ValueError("Lo must be non-negative!")
        if hi <= 0:
            raise ValueError("Hi must be non-negative!")
        if lo > hi:
            raise ValueError("Lo must be smaller than or equal to hi!")

        self.lo = lo
        self.hi = hi

    def __call__(self, image: PilImage, targets: PilImage) -> Tuple[PilImage, PilImage]:
        scale = random_float(self.lo, self.hi)
        new_width = round(scale * image.width)
        new_height = round(scale * image.height)

        image = image.resize((new_width, new_height), PIL.Image.BILINEAR)
        targets = targets.resize((new_width, new_height), PIL.Image.NEAREST)
        return image, targets
