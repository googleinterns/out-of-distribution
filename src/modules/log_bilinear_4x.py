from typing import List

import torch


def log_bilinear_4x(x: torch.Tensor) -> torch.Tensor:
    batch_size, n_channels, height, width = x.size()
    new_height = height * 4
    new_width = width * 4

    # [H_out, W_out]
    top_indices = repeat_column(compute_prev_indices(new_height), new_width)
    bottom_indices = repeat_column(compute_next_indices(new_height), new_width)
    top_areas = repeat_column(compute_prev_areas(height), new_width)
    bottom_areas = repeat_column(compute_next_areas(height), new_width)

    # [H_out, W_out]
    left_indices = repeat_row(compute_prev_indices(new_width), new_height)
    right_indices = repeat_row(compute_next_indices(new_width), new_height)
    left_areas = repeat_row(compute_prev_areas(width), new_height)
    right_areas = repeat_row(compute_next_areas(width), new_height)

    # [N, C, H_out, W_out]
    top_left_anchors = spatial_index(x, top_indices, left_indices)
    top_right_anchors = spatial_index(x, top_indices, right_indices)
    bottom_left_anchors = spatial_index(x, bottom_indices, left_indices)
    bottom_right_anchors = spatial_index(x, bottom_indices, right_indices)

    # [N, C, H_out, W_out]
    top_left_values = torch.log(top_areas) + torch.log(left_areas) + top_left_anchors
    top_right_values = torch.log(top_areas) + torch.log(right_areas) + top_right_anchors
    bottom_left_values = torch.log(bottom_areas) + torch.log(left_areas) + bottom_left_anchors
    bottom_right_values = torch.log(bottom_areas) + torch.log(right_areas) + bottom_right_anchors

    # [N, C, H_out, W_out]
    return logsumexp_tensors([top_left_values, top_right_values, bottom_left_values, bottom_right_values])


def compute_prev_indices(new_n: int) -> torch.Tensor:
    new_indices = torch.arange(new_n)
    prev_indices = (new_indices - 2) // 4
    prev_indices[:2] = -1
    return prev_indices


def compute_next_indices(new_n: int) -> torch.Tensor:
    new_indices = torch.arange(new_n)
    next_indices = (new_indices + 2) // 4
    next_indices[-2:] = -1
    return next_indices


def compute_prev_areas(old_n: int) -> torch.Tensor:
    return torch.tensor([0.0, 0.0] + [7/8, 5/8, 3/8, 1/8] * (old_n - 1) + [1.0, 1.0])


def compute_next_areas(old_n: int) -> torch.Tensor:
    return torch.tensor([1.0, 1.0] + [1/8, 3/8, 5/8, 7/8] * (old_n - 1) + [0.0, 0.0])


def repeat_row(row: torch.Tensor, times: int) -> torch.Tensor:
    """
    Repeats the given row the given number of times.
    :param row: the row to repeat. Size [W].
    :param times: the number of repetitions = H.
    :return: the given row repeated the given number of times. Size [H, W].
    """
    return row.unsqueeze(0).repeat(times, 1)


def repeat_column(column: torch.Tensor, times: int) -> torch.Tensor:
    """
    Repeats the given column the given number of times.
    :param column: the column to repeat. Size [H].
    :param times: the number of repetitions = W.
    :return: the given column repeated the given number of times. Size [H, W].
    """
    return column.unsqueeze(1).repeat(1, times)


def spatial_index(tensor: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    if i.size() != j.size():
        raise ValueError("i and j must have the same size!")

    batch_size, n_channels = tensor.shape[:2]
    out_height, out_width = i.size()

    i = i.flatten()   # [H_out * W_out]
    j = j.flatten()   # [H_out * W_out]

    result = tensor[:, :, i, j]   # [N, C, H_out * W_out]
    return result.reshape(batch_size, n_channels, out_height, out_width)   # [N, C, H_out, W_out]


def logsumexp_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    result = torch.stack(tensors, dim=0)
    return torch.logsumexp(result, dim=0)
