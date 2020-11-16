from typing import Tuple, List

from torch import nn
from torch.utils.data import DataLoader

from src.datasets.load_cifar10 import load_cifar10_train, load_cifar10_infer, CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD
from src.datasets.load_imagenet import load_imagenet_train, load_imagenet_infer, IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD
from src.misc.collection_object import DictObject
from src.resnet.resnet_3stage_softmax import ResNet20_Softmax, ResNet32_Softmax, ResNet44_Softmax, ResNet56_Softmax
from src.resnet.resnetv2_3stage_gaussian import ResNet29V2_Gaussian, ResNet47V2_Gaussian, ResNet65V2_Gaussian, \
    ResNet83V2_Gaussian
from src.resnet.resnetv2_3stage_softmax import ResNet29V2_Softmax, ResNet47V2_Softmax, ResNet65V2_Softmax, \
    ResNet83V2_Softmax
from src.resnet.resnetv2_4stage_gaussian import ResNet50V2_Gaussian, ResNet101V2_Gaussian, ResNet152V2_Gaussian
from src.resnet.resnetv2_4stage_softmax import ResNet50V2_Softmax, ResNet101V2_Softmax, ResNet152V2_Softmax

MODEL_FACTORIES = {
    # ResNet 3-stage softmax
    "ResNet20_Softmax": ResNet20_Softmax,
    "ResNet32_Softmax": ResNet32_Softmax,
    "ResNet44_Softmax": ResNet44_Softmax,
    "ResNet56_Softmax": ResNet56_Softmax,

    # ResNet-v2 3-stage softmax
    "ResNet29V2_Softmax": ResNet29V2_Softmax,
    "ResNet47V2_Softmax": ResNet47V2_Softmax,
    "ResNet65V2_Softmax": ResNet65V2_Softmax,
    "ResNet83V2_Softmax": ResNet83V2_Softmax,

    # ResNet-v2 3-stage Gaussian
    "ResNet29V2_Gaussian": ResNet29V2_Gaussian,
    "ResNet47V2_Gaussian": ResNet47V2_Gaussian,
    "ResNet65V2_Gaussian": ResNet65V2_Gaussian,
    "ResNet83V2_Gaussian": ResNet83V2_Gaussian,

    # ResNet-v2 4-stage softmax
    "ResNet50V2_Softmax": ResNet50V2_Softmax,
    "ResNet101V2_Softmax": ResNet101V2_Softmax,
    "ResNet152V2_Softmax": ResNet152V2_Softmax,

    # ResNet-v2 4-stage Gaussian
    "ResNet50V2_Gaussian": ResNet50V2_Gaussian,
    "ResNet101V2_Gaussian": ResNet101V2_Gaussian,
    "ResNet152V2_Gaussian": ResNet152V2_Gaussian
}

DATASET_FACTORIES = {
    "cifar10": (load_cifar10_train, load_cifar10_infer, CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
    "imagenet": (load_imagenet_train, load_imagenet_infer, IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
}


def create_resnet(cfg: DictObject) -> nn.Module:
    factory = MODEL_FACTORIES[cfg.model.architecture]

    if cfg.model.architecture.endswith("Softmax"):
        return factory(cfg.model.n_classes)
    elif cfg.model.architecture.endswith("Gaussian"):
        return factory(cfg.model.radius, cfg.model.n_classes)
    else:
        raise ValueError


def create_dataset(cfg: DictObject) -> Tuple[DataLoader, DataLoader, DataLoader, List[float], List[float]]:
    train_factory, infer_factory, mean, std = DATASET_FACTORIES[cfg.data.dataset]
    train_loader = train_factory(cfg.data.train_batch_size, cfg.data.n_workers)
    infer_train_loader = infer_factory("train", cfg.data.infer_batch_size, cfg.data.n_workers)
    infer_val_loader = infer_factory("val", cfg.data.infer_batch_size, cfg.data.n_workers)
    return train_loader, infer_train_loader, infer_val_loader, mean, std
