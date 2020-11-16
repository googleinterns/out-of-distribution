import torch
from torchvision.models.detection.image_list import ImageList as TorchImageList
from torchvision.models.detection.rpn import AnchorGenerator as TorchAnchorGenerator

from src.faster_rcnn.anchors import AnchorGenerator
from src.misc.test_case import TestCase


class TestAnchors(TestCase):
    def test_anchor_generator(self) -> None:
        image = torch.empty(1, 3, 512, 512)
        features_list = [
            torch.empty(1, 256, 128, 128), torch.empty(1, 256, 64, 64), torch.empty(1, 256, 32, 32),
            torch.empty(1, 256, 16, 16), torch.empty(1, 256, 8, 8)
        ]

        image_list = TorchImageList(image, [image.shape[2:]])
        expected_generator = TorchAnchorGenerator(sizes=[32, 64, 128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0])
        expected = expected_generator(image_list, features_list)
        expected = torch.cat(expected, dim=0)
        expected = expected[:, [0, 2, 1, 3]].int()

        actual_generator = AnchorGenerator([32, 64, 128, 256, 512], [0.5, 1.0, 2.0])
        actual = actual_generator(image, features_list)
        actual = torch.cat(actual, dim=0)

        self.assert_tensors_equal(expected, actual)
