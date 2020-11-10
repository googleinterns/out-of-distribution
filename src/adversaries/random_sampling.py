import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput
from src.misc.utils import random_float_like


class RandomSampling(Adversary):
    """
    Implements a random sampler that attempts to find adversarial examples close to the original images.
    """
    epsilon: float
    n_restarts: int

    def __init__(self, epsilon: float, n_restarts: int):
        self.epsilon = epsilon
        self.n_restarts = n_restarts

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        result = images.clone()
        best_loss = self.compute_objective(model, result, labels, "none")   # [N]

        for _ in range(self.n_restarts):
            perturbed = torch.clamp(images + random_float_like(images, -self.epsilon, self.epsilon), 0, 1)
            loss = self.compute_objective(model, perturbed, labels, "none")   # [N]

            is_more_adversarial = loss > best_loss
            result = torch.where(is_more_adversarial[:, None, None, None], perturbed, result)
            best_loss = torch.max(loss, best_loss)

        return AdversaryOutput(result, result - images)
