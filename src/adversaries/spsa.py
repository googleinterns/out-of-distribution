import torch
from torch import nn

from src.adversaries.adversary import Adversary, AdversaryOutput
from src.misc.utils import clamp_min_tensor, clamp_max_tensor, random_choice_tensor


class SimultaneousPerturbationStochasticApproximation(Adversary):
    """
    Implements the Simultaneous Perturbation Stochastic Approximation method for generating adversarial examples.

    Note: data batch size for finite difference methods must be 1. To show what can go wrong otherwise, consider a case
    with data batch size 2, sample batch size 1, and:
        f(x0 + δ * v0) - f(x0 - δ * v0) = C
        f(x1 + δ * v1) - f(x1 - δ * v1) = -C

    It follows that:
        right = mean{ f(x0 + δ * v0), f(x1 + δ * v1) }
        left = mean{ f(x0 - δ * v0), f(x1 - δ * v1) }
        right - left = 0
        finite_difference = 0

    The two data points' contribution to the finite difference "cancels out", giving the impression of a vanished
    gradient when in fact each data point should have its own gradient.

    Reference papers:
    - Adversarial Risk and the Dangers of Evaluating Against Weak Attacks (https://arxiv.org/pdf/1802.05666.pdf)
    """
    epsilon: float
    perturb_size: float
    step_size: float
    batch_size: int
    n_iters: int

    def __init__(self, epsilon: float, perturb_size: float, step_size: float, batch_size: int, n_iters: int):
        self.epsilon = epsilon
        self.perturb_size = perturb_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_iters = n_iters

    def __call__(self, model: nn.Module, image: torch.Tensor, label: torch.Tensor) -> AdversaryOutput:
        if image.size(0) != 1:
            raise ValueError("Data batch size must be 1!")

        lo = torch.clamp_min(image - self.epsilon, 0)   # [1, C, H, W]
        hi = torch.clamp_max(image + self.epsilon, 1)   # [1, C, H, W]
        result = image.clone()  # [1, C, H, W]

        for _ in range(self.n_iters):
            directions_size = (self.batch_size, *image.shape[1:])
            directions = random_choice_tensor(directions_size, 1.0, -1.0, 0.5, device=result.device)   # [S, C, H, W]

            # repeat result and label across batch dimension
            results = result.repeat(self.batch_size, 1, 1, 1)   # [S, C, H, W]
            labels = label.repeat(self.batch_size)   # [S]

            # we use self.compute_objective to adapt SPSA for untargeted attacks
            left = self.compute_objective(model, results - self.perturb_size * directions, labels, "none")   # [S]
            right = self.compute_objective(model, results + self.perturb_size * directions, labels, "none")   # [S]

            finite_difference = (right - left) / (2 * self.perturb_size)   # [S]
            gradient = finite_difference[:, None, None, None] / directions   # [S, C, H, W]
            gradient = gradient.mean(dim=0)   # [C, H, W]

            result += self.step_size * gradient
            clamp_min_tensor(result, lo)
            clamp_max_tensor(result, hi)

        return AdversaryOutput(result, result - image)
