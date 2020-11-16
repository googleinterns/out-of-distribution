import json
import os
import shutil
from typing import Tuple, List, Callable

import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from root import from_root
from src.experiments.config import create_resnet, create_dataset
from src.misc.collection_object import DictObject
from src.misc.summary_statistics import SummaryStatistics
from src.misc.utils import set_deterministic_seed, model_device, normalize_tensor, write_lines
from src.modules.max_mahalanobis import mmc_loss
from src.optimizers.sgd import SGD
from src.misc.train_tracker import Tracker, Mode
from src.resnet.shared import ResNet_Softmax, ResNet_Gaussian


class Trainer:
    model: nn.Module
    train_loader: DataLoader
    infer_train_loader: DataLoader
    infer_val_loader: DataLoader
    logger: SummaryWriter
    optimizer: SGD
    scheduler: optim.lr_scheduler.MultiStepLR
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    cfg: DictObject

    def __init__(self, cfg_filepath: str):
        set_deterministic_seed()
        with open(from_root(cfg_filepath), "r") as file:
            self.cfg = DictObject(json.load(file))

        for sub_dirname in ("logs", "checkpoints", "debug"):
            os.makedirs(self.from_out(sub_dirname), exist_ok=True)

        self.model = create_resnet(self.cfg)
        self.model = self.model.to(self.cfg.model.device)

        self.train_loader, self.infer_train_loader, self.infer_val_loader, \
            train_mean, train_std = create_dataset(self.cfg)
        self.model.normalize.set_parameters(train_mean, train_std)

        self.logger = SummaryWriter(log_dir=self.from_out("logs"))

        self.optimizer = create_optimizer(self.model, self.logger, self.cfg)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.cfg.scheduler.milestones, gamma=self.cfg.scheduler.gamma)

        if isinstance(self.model, ResNet_Softmax):
            self.criterion = lambda logits, targets: F.nll_loss(logits, targets, reduction="mean")
        else:
            assert isinstance(self.model, ResNet_Gaussian)
            self.criterion = lambda sqr_distances, targets: mmc_loss(sqr_distances, targets, reduction="mean")

        Tracker.reset(self.cfg.optimizer.n_epochs)
        if self.cfg.load_checkpoint is not None:
            self.load_checkpoint(self.cfg.load_checkpoint)

    def run(self) -> None:
        with torch.autograd.set_detect_anomaly(self.cfg.debug.detect_anomaly):
            while Tracker.epoch <= Tracker.n_epochs:
                self.run_epoch()
                Tracker.epoch += 1

    def run_epoch(self) -> None:
        self.model.train()
        Tracker.mode = Mode.TRAINING
        self.train_epoch()

        self.model.eval()
        Tracker.mode = Mode.INFER_TRAIN
        train_data_accuracy = self.infer_epoch()

        Tracker.mode = Mode.INFER_VAL
        val_data_accuracy = self.infer_epoch()

        self.logger.add_scalars("accuracy", {
            "train": train_data_accuracy,
            "val": val_data_accuracy
        }, global_step=Tracker.epoch)

        self.logger.flush()
        self.scheduler.step()

        if self.cfg.debug.visualize_filters is not None:
            filters = self.model.state_dict()[self.cfg.debug.visualize_filters].detach().clone()
            filters = normalize_tensor(filters)
            torchvision.utils.save_image(filters, self.from_out(f"debug/filters_epoch_{Tracker.epoch}.png"))

        if val_data_accuracy > Tracker.best_val_accuracy:
            Tracker.best_val_accuracy = val_data_accuracy
            self.save_checkpoint(True)
        else:
            self.save_checkpoint(False)

    def train_epoch(self) -> None:
        Tracker.n_batches = len(self.train_loader)
        for Tracker.batch, (images, labels) in enumerate(self.train_loader, 1):
            self.train_batch(images, labels)
            Tracker.global_batch += 1

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images = images.to(model_device(self.model))
        labels = labels.to(model_device(self.model))

        self.optimizer.zero_grad()
        if Tracker.global_batch <= self.cfg.debug.visualize_inputs:
            torchvision.utils.save_image(images, self.from_out(f"debug/inputs_batch_{Tracker.global_batch}.png"))
            lines = [str(label) for label in labels.tolist()]
            write_lines(self.from_out(f"debug/inputs_batch_{Tracker.global_batch}.txt"), lines)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.logger.add_scalar("loss", loss.item(), global_step=Tracker.global_batch)
        Tracker.progress()

    def infer_epoch(self) -> float:
        total_correct = 0
        total_incorrect = 0

        loader = {
            Mode.INFER_TRAIN: self.infer_train_loader,
            Mode.INFER_VAL: self.infer_val_loader
        }[Tracker.mode]

        with torch.no_grad():
            Tracker.n_batches = len(loader)
            for Tracker.batch, (images, labels) in enumerate(loader, 1):
                batch_correct, batch_incorrect = self.infer_batch(images, labels)
                total_correct += batch_correct
                total_incorrect += batch_incorrect

        return total_correct / (total_correct + total_incorrect)

    def infer_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
        images = images.to(model_device(self.model))
        labels = labels.to(model_device(self.model))

        log_probs = self.model(images)
        predictions = torch.argmax(log_probs, dim=1)

        n_correct = torch.sum(labels == predictions).item()
        n_incorrect = labels.size(0) - n_correct

        Tracker.progress()
        return n_correct, n_incorrect

    def load_checkpoint(self, epoch: int) -> None:
        checkpoint_path = self.from_out(f"checkpoints/checkpoint_{epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=model_device(self.model))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        Tracker.epoch = checkpoint["epoch"] + 1
        Tracker.global_batch = checkpoint["global_batch"]
        Tracker.best_val_accuracy = checkpoint["best_val_accuracy"]

        shutil.rmtree(self.from_out("logs"))
        shutil.copytree(self.from_out(f"checkpoints/logs_{epoch}"), self.from_out("logs"))

    def save_checkpoint(self, is_best_epoch: bool) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": Tracker.epoch,
            "global_batch": Tracker.global_batch,
            "best_val_accuracy": Tracker.best_val_accuracy
        }

        torch.save(checkpoint, self.from_out(f"checkpoints/checkpoint_{Tracker.epoch}.pth"))
        if is_best_epoch:
            with open(self.from_out("checkpoints/best_epoch.txt"), "w") as file:
                file.write(str(Tracker.epoch))

        if os.path.isdir(self.from_out(f"checkpoints/logs_{Tracker.epoch}")):
            shutil.rmtree(self.from_out(f"checkpoints/logs_{Tracker.epoch}"))
        shutil.copytree(self.from_out("logs"), self.from_out(f"checkpoints/logs_{Tracker.epoch}"))

    def from_out(self, relative_path: str) -> str:
        return os.path.join(from_root(self.cfg.out_dirpath), relative_path)


def create_optimizer(model: nn.Module, logger: SummaryWriter, cfg: DictObject) -> SGD:
    weights = {"params": [], "weight_decay": cfg.optimizer.weight_decay}
    others = {"params": [], "weight_decay": 0}

    for name, param in model.named_parameters():
        if name.startswith("bn") or ".bn" in name:
            others["params"].append(param)
        elif ".bias" in name:
            others["params"].append(param)
        else:
            weights["params"].append(param)

    def learning_rates_hook(learning_rates: List[float]) -> None:
        summary_statistics = SummaryStatistics(learning_rates)
        logger.add_scalar("learning_rate", summary_statistics.mean, global_step=Tracker.global_batch)

    def updates_to_weights_hook(updates_to_weights: List[float]) -> None:
        summary_statistics = SummaryStatistics(updates_to_weights)
        logger.add_scalars("updates_to_weights", {
            "min": summary_statistics.min,
            "mean": summary_statistics.mean,
            "median": summary_statistics.median,
            "max": summary_statistics.max
        }, global_step=Tracker.global_batch)

    def regularization_to_gradients_hook(regularization_to_gradients: List[float]) -> None:
        summary_statistics = SummaryStatistics(regularization_to_gradients)
        logger.add_scalars("regularization_to_gradients", {
            "min": summary_statistics.min,
            "mean": summary_statistics.mean,
            "median": summary_statistics.median,
            "max": summary_statistics.max
        }, global_step=Tracker.global_batch)

    optimizer = SGD(
        [weights, others], lr=cfg.optimizer.learning_rate, momentum=cfg.optimizer.momentum,
        nesterov=cfg.optimizer.nesterov
    )
    optimizer.learning_rates_hooks.add(learning_rates_hook)
    optimizer.updates_to_weights_hooks.add(updates_to_weights_hook)
    optimizer.regularization_to_gradients_hooks.add(regularization_to_gradients_hook)

    return optimizer
