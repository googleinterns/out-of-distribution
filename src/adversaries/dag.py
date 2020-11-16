from typing import Dict, Any

import torch
from torch import nn


class DenseAdversaryGeneration:
    def __init__(self, max_iters: int):
        self.max_iters = max_iters

    def __call__(self, pipeline: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """
        Adversarially perturbs the given images.
        :param images: the original images. Size [N, C, H, W].
        :param labels: the ground-truth pixel-wise labels. Size [N, H, W].
        :return: the perturbed images and the noise (TODO)
        """
        result = images.clone()   # [N, C, H, W]
        result.requires_grad = True

        for _ in range(self.max_iters):
            if result.grad is not None:
                result.grad.detach_()
                result.grad.zero_()

            logits = pipeline(result, labels)   # [N, K, H, W]
            targets = torch.argmax(logits, dim=1) == labels   # [N, H, W]

            loss = torch.sum()
            loss = torch.sum(loss)
            targets.backward()
