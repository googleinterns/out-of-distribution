import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput
from src.misc.utils import clamp_min_tensor, clamp_max_tensor


class IterativeFgsm(Adversary):
    """
    Implements the Iterative FGSM method for generating adversarial examples.

    Reference papers:
    - Adversarial Machine Learning at Scale (https://arxiv.org/pdf/1611.01236.pdf)
    """
    epsilon: float
    step_size: float

    def __init__(self, epsilon: float, step_size: float):
        self.epsilon = epsilon
        self.step_size = step_size

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        pixel_epsilon = self.epsilon * 255
        n_iters = round(min(pixel_epsilon + 4, 1.25 * pixel_epsilon))   # according to the policy in the reference paper

        lo = torch.clamp_min(images - self.epsilon, 0)
        hi = torch.clamp_max(images + self.epsilon, 1)

        result = images.clone()
        result.requires_grad = True

        for _ in range(n_iters):
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
