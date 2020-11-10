from dataclasses import dataclass
from math import inf
from typing import Tuple

import numpy as np
import torch
from torch import nn

from src.adversaries.adversary import AdversaryOutput, Adversary
from src.misc.utils import atanh, spatial_norm


@dataclass
class CarliniAndWagnerOutput(AdversaryOutput):
    l2_norm: torch.Tensor   # [N]


class CarliniAndWagner(Adversary):
    """
    Implements the Carlini and Wagner method for generating adversarial examples.

    Reference papers:
    - Towards Evaluating the Robustness of Neural Networks (https://arxiv.org/pdf/1608.04644.pdf)

    Reference implementations:
    - CleverHans (https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py)
    """
    n_search_steps: int
    n_iters_per_search: int
    initial_constant: float
    learning_rate: float

    def __init__(self, n_search_steps: int, n_iters_per_search: int, initial_constant: float, learning_rate: float):
        self.n_search_steps = n_search_steps
        self.n_iters_per_search = n_iters_per_search
        self.initial_constant = initial_constant
        self.learning_rate = learning_rate

    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> CarliniAndWagnerOutput:
        batch_size = images.size(0)

        # Loop invariants:
        # - the search space of possible values of c is (lo, hi]
        # - the current value of c we are testing is mid
        # - the lowest-L2 adversarial perturbation is best_delta, and its L2-norm is best_norm
        lo = torch.full([batch_size], 0.0, device=images.device)   # [N]
        hi = torch.full([batch_size], inf, device=images.device)   # [N]
        mid = torch.full([batch_size], self.initial_constant, device=images.device)   # [N]
        best_delta = torch.full_like(images, np.nan)   # [N, C, H, W]
        best_norm = torch.full([batch_size], inf, device=images.device)   # [N]

        for _ in range(self.n_search_steps):
            is_adversarial, delta = self.test_c(model, images, labels, mid)   # [N], [N, C, H, W]
            is_hi_inf: torch.Tensor = hi == inf   # [N]   save this before making any changes to hi

            norm = spatial_norm(delta)   # [N]
            is_norm_smaller = norm < best_norm   # [N]

            # Update logic:
            #   if is_adversarial:
            #     hi = mid
            #     mid = (lo + hi) / 2
            #     if norm < best_norm:
            #       best_norm = norm
            #       best_delta = delta
            #   else:
            #     lo = mid
            #     mid = mid * 10 if hi == inf else (lo + hi) / 2

            # Case 1a: is_adversarial and norm < best_norm
            hi_next1 = mid
            mid_next1 = (lo + hi_next1) / 2
            lo_next1 = lo   # lo remains unchanged
            best_delta_next1a = delta
            best_norm_next1a = norm

            # Case 1b: is_adversarial and norm >= best_norm
            best_delta_next1b = best_delta   # best_delta remains unchanged
            best_norm_next1b = best_norm   # best_norm remains unchanged

            # Case 2a: not is_adversarial and hi == inf
            lo_next2 = mid
            mid_next2a = mid * 10   # x10 rule is from CleverHans implementation
            hi_next2 = hi   # hi remains unchanged
            best_delta_next2 = best_delta   # best_delta remains unchanged
            best_norm_next2 = best_norm   # best_norm remains unchanged

            # Case 2b: not is_adversarial and hi != inf
            mid_next2b = (lo_next2 + hi) / 2

            lo = torch.where(is_adversarial, lo_next1, lo_next2)
            hi = torch.where(is_adversarial, hi_next1, hi_next2)
            mid = torch.where(is_adversarial, mid_next1, torch.where(is_hi_inf, mid_next2a, mid_next2b))
            best_delta = torch.where(
                is_adversarial[:, None, None, None],
                torch.where(is_norm_smaller[:, None, None, None], best_delta_next1a, best_delta_next1b),
                best_delta_next2
            )
            best_norm = torch.where(
                is_adversarial,
                torch.where(is_norm_smaller, best_norm_next1a, best_norm_next1b),
                best_norm_next2
            )

        return CarliniAndWagnerOutput(images + best_delta, best_delta, best_norm)

    def test_c(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, c: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perturbs the given images to minimize the objective function with the given value of c.
        :param model: the model to attack
        :param images: the images to perturb
        :param labels: the ground-truth labels of the given images
        :param c: the value of c to use in the objective function
        :return:
        - whether each image was successfully adversarially perturbed. Size [N].
        - the final noise. Size [N, C, H, W].
        """
        w = atanh(2 * images - 1)   # [N, C, H, W]
        w = torch.autograd.Variable(w, requires_grad=True)

        def objective(w: torch.autograd.Variable) -> torch.Tensor:
            perturbed_images = (torch.tanh(w) + 1) / 2   # [N, C, H, W]
            delta = perturbed_images - images   # [N, C, H, W]
            norm = spatial_norm(delta)   # [N]

            # we use -self.compute_objective to adapt C&W for untargeted attacks
            f = -self.compute_objective(model, perturbed_images, labels, "none")   # [N]
            return torch.mean(norm + c * f)

        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        for _ in range(self.n_iters_per_search):
            optimizer.zero_grad()
            objective_val = objective(w)
            objective_val.backward()
            optimizer.step()

        with torch.no_grad():
            perturbed_images = (torch.tanh(w) + 1) / 2   # [N, C, H, W]
            delta = perturbed_images - images   # [N, C, H, W]
            f = -self.compute_objective(model, perturbed_images, labels, "none")   # [N]
            return f < 0, delta
