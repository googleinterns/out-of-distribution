import os

import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from root import from_root
from src.misc.test_case import TestCase
from src.modules.log_bilinear_4x import log_bilinear_4x


class TestLogBilinear4x(TestCase):
    def test_linear_domain(self) -> None:
        images = [
            Image.open(from_root("test_data/log_bilinear_4x/1.png")),
            Image.open(from_root("test_data/log_bilinear_4x/2.png"))
        ]
        transform = transforms.ToTensor()
        x = torch.stack([transform(image) for image in images], dim=0)

        os.makedirs(from_root("test_results/log_bilinear_4x"), exist_ok=True)

        expected = F.interpolate(x, scale_factor=4, mode="bilinear")
        torchvision.utils.save_image(expected, from_root("test_results/log_bilinear_4x/expected.png"))

        actual = torch.log(x)
        actual = log_bilinear_4x(actual)
        actual = torch.exp(actual)
        torchvision.utils.save_image(actual, from_root("test_results/log_bilinear_4x/actual.png"))

        self.assert_no_nan(actual)
        self.assert_tensors_almost_equal(expected, actual, 1e-6)

    def test_log_domain(self) -> None:
        images = [
            Image.open(from_root("test_data/log_bilinear_4x/1.png")),
            Image.open(from_root("test_data/log_bilinear_4x/2.png"))
        ]
        transform = transforms.ToTensor()
        x = torch.stack([transform(image) for image in images], dim=0)

        expected = F.interpolate(x, scale_factor=4, mode="bilinear")
        expected = torch.log(expected)

        actual = torch.log(x)
        actual = log_bilinear_4x(actual)

        self.assert_no_nan(actual)
        self.assert_tensors_almost_equal(expected, actual, 1e-6)
