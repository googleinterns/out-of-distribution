from dataclasses import dataclass
from math import log, pi
from typing import Union

import torch
from torch import nn

from src.misc.utils import select_row_items
from src.resnet.shared import GaussianMode


@dataclass
class GaussianResult:
    log_likelihoods: torch.Tensor
    log_posteriors: torch.Tensor


class MaxMahalanobis(nn.Module):
    centers: torch.Tensor
    out_channels: int

    def __init__(self, radius: float, n_features: int, n_classes: int):
        super().__init__()
        self.init_centers(radius, n_features, n_classes)
        self.out_channels = n_classes

    def init_centers(self, radius: float, n_features: int, n_classes: int) -> None:
        centers = torch.zeros((n_classes, n_features))
        centers[0, 0] = 1

        for c in range(1, n_classes):
            for d in range(c):
                numerator = 1 + torch.dot(centers[c, :], centers[d, :]) * (n_classes - 1)
                denominator = centers[d, d] * (n_classes - 1)
                centers[c, d] = -numerator / denominator

            remaining_norm = 1 - torch.norm(centers[c, :]) ** 2
            centers[c, c] = torch.sqrt(torch.abs(remaining_norm))

        centers *= radius
        self.register_buffer("centers", centers)

    def forward(self, x: torch.Tensor, mode: GaussianMode) -> Union[torch.Tensor, GaussianResult]:
        differences = x[:, None, :] - self.centers   # [N, K, D]
        sqr_distances = torch.norm(differences, dim=2) ** 2   # [N, K]

        if mode == GaussianMode.SQR_DISTANCES:
            return sqr_distances

        d = x.size(1)
        norm_constant = d/2 * log(2 * pi)
        log_likelihoods = -norm_constant - 1/2 * sqr_distances   # [N, K]

        if mode == GaussianMode.LOG_LIKELIHOODS:
            return log_likelihoods

        log_denominator = torch.logsumexp(log_likelihoods, dim=1)   # [N]
        log_posteriors = log_likelihoods - log_denominator[:, None]   # [N, K]

        if mode == GaussianMode.LOG_POSTERIORS:
            return log_posteriors

        assert mode == GaussianMode.BOTH
        return GaussianResult(log_likelihoods, log_posteriors)


def mmc_loss(sqr_distances: torch.Tensor, labels: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction not in {"none", "mean"}:
        raise ValueError("Reduction must be 'none' or 'mean'!")

    result = select_row_items(sqr_distances, labels)   # TODO: make this / 2
    return result if reduction == "none" else result.mean()
