import multiprocessing
import os
from typing import Union

import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from root import from_root
from src.misc.utils import read_lines
from src.modules.color import color

DATA_DIRPATH = from_root("data/mnist")
SPLIT_DIRPATH = from_root("splits/mnist")


def load_mnist_infer(split: str, batch_size: int, n_workers: Union[str, int]) -> DataLoader:
    if split not in {"train", "val", "test"}:
        raise ValueError("Split must be 'train', 'val', or 'test'!")
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.Lambda(color),
        transforms.ToTensor()
    ])

    if split == "train":
        dataset = torchvision.datasets.MNIST(DATA_DIRPATH, train=True, transform=transform, download=True)
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), int)
        dataset = Subset(dataset, indices)
    elif split == "val":
        dataset = torchvision.datasets.MNIST(DATA_DIRPATH, train=True, transform=transform, download=True)
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "val.txt"), int)
        dataset = Subset(dataset, indices)
    else:
        dataset = torchvision.datasets.MNIST(DATA_DIRPATH, train=False, transform=transform, download=True)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
