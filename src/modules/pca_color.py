from typing import List

import torch


class PcaColor:
    std: float
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor

    def __init__(self, std: float, eigenvectors: List[List[float]], eigenvalues: List[float]):
        if std < 0:
            raise ValueError("Std must be non-negative!")

        self.std = std
        self.eigenvectors = torch.tensor(eigenvectors)
        self.eigenvalues = torch.tensor(eigenvalues)

        if self.eigenvectors.shape != (3, 3):
            raise ValueError("Eigenvectors must have size [C=3, K=3]!")
        if self.eigenvalues.size() != (3,):
            raise ValueError("Eigenvalues must have size [K=3]!")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.size(0) != 3:
            raise ValueError("Image must have 3 channels!")

        multiples = torch.normal(0, self.std, [3])   # [K]
        deltas = self.eigenvectors * multiples * self.eigenvalues   # [C, K]
        deltas = torch.sum(deltas, dim=1)   # [C]

        result = image + deltas[:, None, None]   # [C, H, W]
        result.clamp_(0, 1)   # [C, H, W]
        return result
