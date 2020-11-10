import multiprocessing
import os
from typing import Union

import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms

from root import from_root
from src.misc.utils import read_lines

DATA_DIRPATH = from_root("data/svhn")
SPLIT_DIRPATH = from_root("splits/svhn")
SVHN_TRAIN_MEAN = [0.4310, 0.4303, 0.4464]
SVHN_TRAIN_STD = [0.1965, 0.1983, 0.1994]


def load_svhn_infer(split: str, batch_size: int, n_workers: Union[str, int]) -> DataLoader:
    if split not in {"train", "val", "test"}:
        raise ValueError("Split must be 'train', 'val', or 'test'!")
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.ToTensor()

    if split == "train":
        dataset = ConcatDataset([
            torchvision.datasets.SVHN(DATA_DIRPATH, split="train", transform=transform, download=True),
            torchvision.datasets.SVHN(DATA_DIRPATH, split="extra", transform=transform, download=True)
        ])
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), int)
        dataset = Subset(dataset, indices)
    elif split == "val":
        dataset = ConcatDataset([
            torchvision.datasets.SVHN(DATA_DIRPATH, split="train", transform=transform, download=True),
            torchvision.datasets.SVHN(DATA_DIRPATH, split="extra", transform=transform, download=True)
        ])
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "val.txt"), int)
        dataset = Subset(dataset, indices)
    else:
        dataset = torchvision.datasets.SVHN(DATA_DIRPATH, split="test", transform=transform, download=True)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
