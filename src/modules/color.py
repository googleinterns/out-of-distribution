from PIL.Image import Image as PilImage


def color(image: PilImage) -> PilImage:
    return image.convert("RGB")
