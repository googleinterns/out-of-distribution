import multiprocessing
import os
from typing import Union

import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from root import from_root
from src.misc.utils import read_lines

DATA_DIRPATH = from_root("data/cifar10")
SPLIT_DIRPATH = from_root("splits/cifar10")
CIFAR10_TRAIN_MEAN = [0.4913, 0.4820, 0.4464]
CIFAR10_TRAIN_STD = [0.2470, 0.2434, 0.2616]


def load_cifar10_train(batch_size: int, n_workers: Union[str, int]) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=True, transform=transform, download=True)
    train_indices = read_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), int)
    train_dataset = Subset(dataset, train_indices)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)


def load_cifar10_infer(split: str, batch_size: int, n_workers: Union[str, int]) -> DataLoader:
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
        dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=True, transform=transform, download=True)
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), int)
        dataset = Subset(dataset, indices)
    elif split == "val":
        dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=True, transform=transform, download=True)
        indices = read_lines(os.path.join(SPLIT_DIRPATH, "val.txt"), int)
        dataset = Subset(dataset, indices)
    else:
        dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=False, transform=transform, download=True)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
