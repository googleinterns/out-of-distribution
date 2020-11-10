from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from src.misc.utils import min_tensor, select_row_items, remove_row_items
from src.resnet.shared import ResNet_Softmax, ResNet_Gaussian, SoftmaxMode, GaussianMode


@dataclass
class AdversaryOutput:
    perturbed_images: torch.Tensor   # [N, C, H, W]
    noises: torch.Tensor   # [N, C, H, W]


class Adversary:
    @abstractmethod
    def __call__(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> AdversaryOutput:
        """
        Perturbs the given images to adversarially attack the given model.
        :param model: the model to attack
        :param images: the images to perturb
        :param labels: the ground-truth labels of the given images
        :return: the perturbed images and noise
        """
        raise NotImplementedError

    def compute_objective(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor, reduction: str) \
            -> torch.Tensor:
        """
        Objective function to maximize when performing an untargeted adversarial attack.
        :param model: the model to attack
        :param images: the images to perturb
        :param labels: the ground-truth labels of the given images
        :param reduction: the reduction (none or mean) to apply to the output
        :return: the value of the objective function at the given images
        """
        if reduction not in {"none", "mean"}:
            raise ValueError("Reduction must be 'none' or 'mean'!")

        if isinstance(model, ResNet_Softmax):
            logits = model(images, SoftmaxMode.LOGITS)   # [N, K]
            true_logits = select_row_items(logits, labels)   # [N]
            other_logits = remove_row_items(logits, labels)   # [N, K - 1]
            margins = torch.max(other_logits, dim=1)[0] - true_logits   # [N]
            result = min_tensor(1e-8, margins)   # [N]

        else:
            assert isinstance(model, ResNet_Gaussian)
            sqr_distances = model(images, GaussianMode.SQR_DISTANCES)   # [N, K]
            true_sqr_distances = select_row_items(sqr_distances, labels)   # [N]
            other_sqr_distances = remove_row_items(sqr_distances, labels)   # [N, K - 1]
            margins = true_sqr_distances - torch.min(other_sqr_distances, dim=1)[0]   # [N]
            result = min_tensor(1e-8, margins)   # [N]

        return result if reduction == "none" else result.mean()
