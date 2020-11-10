import os
import random

import torchvision

from root import from_root
from src.misc.utils import set_deterministic_seed, write_lines

DATA_DIRPATH = from_root("data/mnist")
SPLIT_DIRPATH = from_root("splits/mnist")
TRAIN_SIZE = 50000


def main() -> None:
    set_deterministic_seed()

    dataset = torchvision.datasets.MNIST(DATA_DIRPATH, train=True, download=True)
    size = len(dataset)

    indices = list(range(size))
    random.shuffle(indices)

    train_indices = indices[:TRAIN_SIZE]
    write_lines(os.path.join(SPLIT_DIRPATH, "train.txt"), train_indices)

    val_indices = indices[TRAIN_SIZE:]
    write_lines(os.path.join(SPLIT_DIRPATH, "val.txt"), val_indices)


if __name__ == "__main__":
    main()
