import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput
from src.misc.utils import random_float_like, clamp_min_tensor, clamp_max_tensor


class ProjectedGradientDescent(Adversary):
    """
    Implements the Projected Gradient Descent method for generating adversarial examples.

    Reference papers:
    - Towards Deep Learning Models Resistant to Adversarial Attacks (https://arxiv.org/pdf/1706.06083.pdf)
    """
    epsilon: float
    n_iters: int
    step_size: float

    def __init__(self, epsilon: float, n_iters: int, step_size: float):
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.step_size = step_size

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        lo = torch.clamp_min(images - self.epsilon, 0)
        hi = torch.clamp_max(images + self.epsilon, 1)

        result = torch.clamp(images + random_float_like(images, -self.epsilon, self.epsilon), 0, 1)
        result.requires_grad = True

        for _ in range(self.n_iters):
            if result.grad is not None:
                result.grad.detach_()
                result.grad.zero_()

            loss = self.compute_objective(model, result, labels, "mean")
            loss.backward()

            with torch.no_grad():
                result += self.step_size * torch.sign(result.grad)
                clamp_min_tensor(result, lo)
                clamp_max_tensor(result, hi)

        result.requires_grad = False
        return AdversaryOutput(result, result - images)
