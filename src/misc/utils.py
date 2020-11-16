import os
import random
from math import pi
from typing import Any, List, Callable, Tuple, Iterator, Reversible, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def atanh(x: torch.Tensor) -> torch.Tensor:
    return (torch.log(1 + x) - torch.log(1 - x)) / 2


def clamp_min_parameter(parameter: nn.Parameter, min_value: float) -> None:
    parameter.data = parameter.data.clamp_min(min_value)


def clamp_min_tensor(tensor: torch.Tensor, lo: torch.Tensor) -> None:
    tensor[tensor < lo] = lo[tensor < lo]


def clamp_max_tensor(tensor: torch.Tensor, hi: torch.Tensor) -> None:
    tensor[tensor > hi] = hi[tensor > hi]


def covariance(x: torch.Tensor) -> torch.Tensor:
    x = x.numpy()
    cov = np.cov(x, rowvar=False, bias=True)
    return torch.from_numpy(cov)


def dot_products(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot product between each pair of rows in the given Tensor.
    :param tensor: the Tensor to compute the dot products between the rows of. Size [N, D].
    :return: the computed dot products, where result[i, j] = tensor[i, :] * tensor[j, :]. Size [N, N].
    """
    products = tensor[:, None, :] * tensor[None, :, :]   # [N, N, D]
    return torch.sum(products, dim=2)   # [N, N]


def euclidean_distances(tensor: torch.Tensor) -> torch.Tensor:
    differences = tensor[:, None, :] - tensor[None, :, :]   # [N, N, D]
    return torch.sqrt(torch.sum(differences ** 2, dim=2))   # [N, N]


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def radians_to_degrees(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 180 / pi


def random_bool(true_prob: float) -> torch.Tensor:
    return random.random() < true_prob


def random_bool_tensor(size: Tuple[int, ...], true_prob: float, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.rand(size, device=device) < true_prob


def random_bool_like(tensor: torch.Tensor, true_prob: float) -> torch.Tensor:
    return torch.rand_like(tensor) < true_prob


def random_choice_tensor(
    size: Tuple[int, ...], choice0: Any, choice1: Any, prob0: float, device: Optional[torch.device] = None
) -> torch.Tensor:
    mask = random_bool_tensor(size, prob0, device=device)
    choice0 = torch.tensor(choice0, device=device)
    choice1 = torch.tensor(choice1, device=device)
    return torch.where(mask, choice0, choice1)


def random_choice_like(
    tensor: torch.Tensor, choice0: Union[int, float], choice1: Union[int, float], prob0: float
) -> torch.Tensor:
    mask = random_bool_like(tensor, prob0)
    choice0 = torch.tensor(choice0, device=tensor.device)
    choice1 = torch.tensor(choice1, device=tensor.device)
    return torch.where(mask, choice0, choice1)


def random_float(lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * random.random()


def random_float_like(tensor: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand_like(tensor)


def max_tensor(value: float, tensor: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(value, dtype=torch.float, device=tensor.device), tensor)


def min_tensor(value: float, tensor: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.tensor(value, dtype=torch.float, device=tensor.device), tensor)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_value = tensor.min()
    max_value = tensor.max()

    if min_value == max_value:
        return torch.full_like(tensor, 0)   # prevent division by zero

    return (tensor - min_value) / (max_value - min_value)


def read_lines(filepath: str, constructor: Callable[[str], Any]) -> List[Any]:
    with open(filepath, "r") as file:
        lines = file.readlines()

    return [constructor(line[:-1]) for line in lines]


def remove_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(tensor.size(0), dtype=torch.bool)
    return tensor.masked_select(mask)


def remove_outliers(df: pd.DataFrame, data_col: str, group_col: str, min_q: float = 0, max_q: float = 100) \
        -> pd.DataFrame:
    """
    TODO
    :param df:
    :param data_col:
    :param group_col:
    :param min_q:
    :param max_q:
    :return:
    """
    builder = []
    for group, subset in df.groupby(group_col):
        min_p = np.percentile(subset[data_col], min_q)
        max_p = np.percentile(subset[data_col], max_q)

        mask = (min_p <= subset[data_col]) & (subset[data_col] <= max_p)
        builder.append(subset[mask])
    return pd.concat(builder, axis=0)


def remove_row_items(tensor: torch.Tensor, column_indices: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = tensor.size()

    mask = torch.full((n_rows, n_cols), True, dtype=torch.bool, device=tensor.device)
    row_indices = torch.arange(n_rows)
    mask[row_indices, column_indices] = False

    result = tensor.masked_select(mask)
    return result.reshape(n_rows, n_cols - 1)


def reversed_zip(*lists: Union[Reversible[Any], nn.ModuleList]) -> Iterator[Tuple[Any]]:
    return zip(reversed(a_list) for a_list in lists)


def select_row_items(tensor: torch.Tensor, column_indices: torch.Tensor) -> torch.Tensor:
    row_indices = torch.arange(tensor.size(0))
    return tensor[row_indices, column_indices]


def set_deterministic_seed() -> None:
    random.seed(1731)
    np.random.seed(1731)
    torch.manual_seed(1731)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed(1731)

    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False


def spatial_norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1).norm(p=2, dim=1)


def write_lines(filepath: str, lines: List[Any]) -> None:
    lines = [str(line) + "\n" for line in lines]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        file.writelines(lines)
