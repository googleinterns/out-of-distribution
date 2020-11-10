import multiprocessing
from typing import Union

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from root import from_root
from src.modules.pca_color import PcaColor
from src.modules.random_resize import RandomResize

DATA_DIRPATH = from_root("data/imagenet")
IMAGENET_TRAIN_MEAN = [0.485, 0.456, 0.406]
IMAGENET_TRAIN_STD = [0.229, 0.224, 0.225]

# Reference statistics:
# https://github.com/fastai/imagenet-fast/blob/faa0f9dfc9e8e058ffd07a248724bf384f526fae/imagenet_nv/fastai_imagenet.py#L93
IMAGENET_EIGENVECTORS = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
IMAGENET_EIGENVALUES = [0.2175, 0.0188, 0.0045]


def load_imagenet_train(batch_size: int, n_workers: Union[str, int]) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.Compose([
        RandomResize(256, 480),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        PcaColor(0.1, IMAGENET_EIGENVECTORS, IMAGENET_EIGENVALUES)
    ])
    dataset = torchvision.datasets.ImageNet(DATA_DIRPATH, split="train", transform=transform)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)


def load_imagenet_infer(split: str, batch_size: int, n_workers: Union[str, int]) -> DataLoader:
    if split not in {"train", "val"}:
        raise ValueError("Split must be 'train' or 'val'!")
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    if split == "train":
        dataset = torchvision.datasets.ImageNet(DATA_DIRPATH, split="train", transform=transform)
    else:
        dataset = torchvision.datasets.ImageNet(DATA_DIRPATH, split="val", transform=transform)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
