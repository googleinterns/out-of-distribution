import unittest

import torch

from src.misc.utils import set_deterministic_seed


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_deterministic_seed()

    def assert_no_nan(self, actual: torch.Tensor) -> None:
        has_nan = torch.any(torch.isnan(actual))
        self.assertFalse(has_nan)

    def assert_tensors_almost_equal(self, expected: torch.Tensor, actual: torch.Tensor, atol: float) -> None:
        self.assertTupleEqual(expected.size(), actual.size())
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(torch.allclose(expected, actual, atol=atol))

    def assert_tensors_equal(self, expected: torch.Tensor, actual: torch.Tensor) -> None:
        self.assertTupleEqual(expected.size(), actual.size())
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(torch.all(expected == actual))

    def assert_tensor_almost_constant(self, actual: torch.Tensor, atol: float) -> None:
        self.assertTrue(torch.max(actual) - torch.min(actual) <= atol)
