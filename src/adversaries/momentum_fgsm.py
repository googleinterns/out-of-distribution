import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput


class MomentumFgsm(Adversary):
    """
    Implements the Momentum Iterative FGSM method for generating adversarial examples.

    Reference papers:
    - Boosting Adversarial Attacks With Momentum (https://arxiv.org/pdf/1710.06081.pdf)
    """
    epsilon: float
    n_iters: int
    decay_factor: float

    def __init__(self, epsilon: float, n_iters: int, decay_factor: float):
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.decay_factor = decay_factor

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        step_size = self.epsilon / self.n_iters
        velocity = torch.zeros_like(images)

        result = images.clone()
        result.requires_grad = True

        for _ in range(self.n_iters):
            if result.grad is not None:
                result.grad.detach_()
                result.grad.zero_()

            loss = self.compute_objective(model, result, labels, "mean")
            loss.backward()

            velocity = self.decay_factor * velocity + result.grad / torch.norm(result.grad, p=1)
            with torch.no_grad():
                result += step_size * torch.sign(velocity)
                result.clamp_(0, 1)

        result.requires_grad = False
        return AdversaryOutput(result, result - images)
