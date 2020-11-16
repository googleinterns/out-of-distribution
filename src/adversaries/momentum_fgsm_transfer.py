import json
import os

import torch
from torch import nn

from root import from_root
from src.adversaries.adversary import Adversary, AdversaryOutput
from src.experiments.config import create_resnet
from src.misc.collection_object import DictObject
from src.misc.utils import model_device


class MomentumFgsmTransfer(Adversary):
    """
    Implements the Momentum Iterative FGSM method for generating adversarial examples in the context of black-box
    transfer-based attack, i.e. perturbations are generated on the surrogate model passed in the constructor.
    """
    surrogate_model: nn.Module
    epsilon: float
    n_iters: int
    decay_factor: float

    def __init__(self, surrogate_cfg_filepath: str, epsilon: float, n_iters: int, decay_factor: float):
        self.init_surrogate_model(surrogate_cfg_filepath)
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.decay_factor = decay_factor

    def init_surrogate_model(self, surrogate_cfg_filepath: str) -> None:
        with open(from_root(surrogate_cfg_filepath), "r") as file:
            cfg = DictObject(json.load(file))

        self.surrogate_model = create_resnet(cfg)
        self.surrogate_model = self.surrogate_model.to(cfg.model.device)

        best_epoch_filepath = os.path.join(from_root(cfg.out_dirpath), "checkpoints/best_epoch.txt")
        with open(best_epoch_filepath, "r") as file:
            epoch = int(file.read())

        checkpoint_filepath = os.path.join(from_root(cfg.out_dirpath), f"checkpoints/checkpoint_{epoch}.pth")
        checkpoint = torch.load(checkpoint_filepath, map_location=model_device(self.surrogate_model))
        self.surrogate_model.load_state_dict(checkpoint["model_state_dict"])

        self.surrogate_model.eval()
        for param in self.surrogate_model.parameters():
            param.requires_grad = False

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        step_size = self.epsilon / self.n_iters
        velocity = torch.zeros_like(images)

        result = images.clone()
        result.requires_grad = True

        for _ in range(self.n_iters):
            if result.grad is not None:
                result.grad.detach_()
                result.grad.zero_()

            loss = self.compute_objective(self.surrogate_model, result, labels, "mean")
            loss.backward()

            velocity = self.decay_factor * velocity + result.grad / torch.norm(result.grad, p=1)
            with torch.no_grad():
                result += step_size * torch.sign(velocity)
                result.clamp_(0, 1)

        result.requires_grad = False
        return AdversaryOutput(result, result - images)
