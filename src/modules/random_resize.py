import random

import torchvision
from PIL.Image import Image as PilImage


class RandomResize:
    lo: int
    hi: int

    def __init__(self, lo: int, hi: int):
        if lo <= 0:
            raise ValueError("Lo must be non-negative!")
        if hi <= 0:
            raise ValueError("Hi must be non-negative!")
        if lo > hi:
            raise ValueError("Lo must be smaller than or equal to hi!")

        self.lo = lo
        self.hi = hi

    def __call__(self, image: PilImage) -> PilImage:
        size = random.randint(self.lo, self.hi)
        transform = torchvision.transforms.Resize(size)
        return transform(image)
