import os
import unittest

import torch
import torchvision
from torch.utils.data import Subset

from src.datasets.load_cifar10 import CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD, DATA_DIRPATH, SPLIT_DIRPATH
from src.misc.utils import read_lines


class TestCifar10(unittest.TestCase):
    def setUp(self) -> None:
        transform = torchvision.transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=True, transform=transform, download=True)

        train_indices = read_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), int)
        self.train_dataset = Subset(dataset, train_indices)
        self.images = torch.stack([image for image, label in self.train_dataset], dim=0)

    def test_mean(self) -> None:
        expected_mean = torch.mean(self.images, dim=(0, 2, 3))
        for c in range(3):
            self.assertAlmostEqual(expected_mean[c].item(), CIFAR10_TRAIN_MEAN[c], 4)

    def test_std(self) -> None:
        expected_std = torch.std(self.images, dim=(0, 2, 3))
        for c in range(3):
            self.assertAlmostEqual(expected_std[c].item(), CIFAR10_TRAIN_STD[c], 4)
