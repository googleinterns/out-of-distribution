import os
import random

import torchvision
from torch.utils.data import ConcatDataset

from root import from_root
from src.misc.utils import set_deterministic_seed, write_lines

DATA_DIRPATH = from_root("data/svhn")
SPLIT_DIRPATH = from_root("splits/svhn")
TRAIN_PCT = 0.8


def main() -> None:
    set_deterministic_seed()

    dataset = ConcatDataset([
        torchvision.datasets.SVHN(DATA_DIRPATH, split="train", download=True),
        torchvision.datasets.SVHN(DATA_DIRPATH, split="extra", download=True)
    ])
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
