import os
import random

import torchvision

from root import from_root
from src.misc.utils import set_deterministic_seed, write_lines

DATA_DIRPATH = from_root("data/cifar10")
SPLIT_DIRPATH = from_root("splits/cifar10")
TRAIN_PCT = 0.9


def main() -> None:
    set_deterministic_seed()

    dataset = torchvision.datasets.CIFAR10(DATA_DIRPATH, train=True, download=True)
    size = len(dataset)

    indices = list(range(size))
    random.shuffle(indices)

    train_size = round(TRAIN_PCT * size)
    train_indices = indices[:train_size]
    write_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), train_indices)

    val_indices = indices[train_size:]
    write_lines(os.path.join(SPLIT_DIRPATH, "val.txt"), val_indices)


if __name__ == "__main__":
    main()
