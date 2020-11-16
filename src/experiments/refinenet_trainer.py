import json
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from root import from_root
from src.datasets.load_pascal_voc import load_pascal_voc_train, load_pascal_voc_infer
from src.misc.collection_object import DictObject
from src.misc.utils import set_deterministic_seed, model_device
from src.optimizers.sgd import SGD
from src.refinenet.refinenet_4cascaded import RefineNet_4Cascaded


class Trainer:
    model: RefineNet_4Cascaded
    train_loader: DataLoader
    infer_train_loader: DataLoader
    infer_val_loader: DataLoader
    logger: SummaryWriter
    optimizer: SGD
    scheduler: optim.lr_scheduler.MultiStepLR
    cfg: DictObject

    def __init__(self, cfg_filepath: str):
        set_deterministic_seed()
        with open(from_root(cfg_filepath), "r") as file:
            self.cfg = DictObject(json.load(file))

        for sub_dirname in ("logs", "checkpoints", "debug"):
            os.makedirs(self.from_out(sub_dirname), exist_ok=True)

        self.model = RefineNet_4Cascaded()
        self.model = self.model.cuda()
        state_dict = torch.load("TODO", map_location=model_device(self.model))
        self.model.backbone.load_state_dict(state_dict)

        self.train_loader = load_pascal_voc_train(16)
        self.infer_train_loader = load_pascal_voc_infer("train", 16)
        self.infer_val_loader = load_pascal_voc_infer("val", 16)
