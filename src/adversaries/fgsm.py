import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput


class FastGradientSignMethod(Adversary):
    """
    Implements the Fast Gradient Sign Method for generating adversarial examples.

    Reference papers:
    - Explaining and Harnessing Adversarial Examples (https://arxiv.org/pdf/1412.6572.pdf)
    """
    epsilon: float

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        result = images.clone()
        result.requires_grad = True

        if result.grad is not None:
            result.grad.detach_()
            result.grad.zero_()

        loss = self.compute_objective(model, result, labels, "mean")
        loss.backward()

        with torch.no_grad():
            result += self.epsilon * torch.sign(result.grad)
            result.clamp_(0, 1)

        result.requires_grad = False
        return AdversaryOutput(result, result - images)
