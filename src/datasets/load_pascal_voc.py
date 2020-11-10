import multiprocessing
from typing import Union

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from root import from_root

from src.modules.random_crops import RandomCrops
from src.modules.random_horizontal_flips import RandomHorizontalFlips
from src.modules.random_scales import RandomScales
from src.modules.to_tensors import ToTensors

# Official dataset website and Oxford's mirror both go down frequently. Therefore, we use Joseph Redmon's mirror at
# https://pjreddie.com/projects/pascal-voc-dataset-mirror/.
DATA_DIRPATH = from_root("data/pascal_voc")


def load_pascal_voc_train(n_workers: Union[str, int]) -> DataLoader:
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = transforms.Compose([
        RandomScales(0.7, 1.3),
        RandomHorizontalFlips(0.5),
        RandomCrops(400),
        ToTensors()
    ])
    dataset = torchvision.datasets.VOCSegmentation(DATA_DIRPATH, image_set="train", transforms=transform)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=n_workers)


def load_pascal_voc_infer(split: str, n_workers: Union[str, int]) -> DataLoader:
    if split not in {"train", "val"}:
        raise ValueError("Split must be 'train' or 'val'!")
    if type(n_workers) == str and n_workers != "n_cores":
        raise ValueError("If n_workers is a string, it must be 'n_cores'!")
    if type(n_workers) == int and n_workers < 0:
        raise ValueError("If n_workers is an int, it must be non-negative!")

    transform = ToTensors()

    if split == "train":
        dataset = torchvision.datasets.VOCSegmentation(DATA_DIRPATH, image_set="train", transforms=transform)
    else:
        dataset = torchvision.datasets.VOCSegmentation(DATA_DIRPATH, image_set="val", transforms=transform)

    if n_workers == "n_cores":
        n_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=n_workers)
