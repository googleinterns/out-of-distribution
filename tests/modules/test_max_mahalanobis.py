import torch

from src.misc.test_case import TestCase
from src.misc.utils import remove_diagonal, dot_products, radians_to_degrees, euclidean_distances
from src.modules.max_mahalanobis import MaxMahalanobis


class TestResNet20_Gaussian(TestCase):
    def test_256_10_norm(self) -> None:
        layer = MaxMahalanobis(10, 256, 10)

        expected_norm = torch.full([10], 10.0)
        actual_norm = torch.norm(layer.centers, dim=1)
        self.assert_tensors_almost_equal(expected_norm, actual_norm, 1e-8)

    def test_256_10_angles(self) -> None:
        layer = MaxMahalanobis(10, 256, 10)

        dot_values = remove_diagonal(dot_products(layer.centers))   # [N choose 2]
        angles = torch.acos(dot_values / 10 ** 2)   # [N choose 2]
        angles = radians_to_degrees(angles)   # [N choose 2]

        self.assert_tensor_almost_constant(angles, 1e-5)
        print(angles[0])

    def test_256_10_distances(self) -> None:
        layer = MaxMahalanobis(10, 256, 10)

        dist_values = remove_diagonal(euclidean_distances(layer.centers))   # [N choose 2]
        print(dist_values.min())
        print(dist_values.max())

    def test_2048_1000_angles(self) -> None:
        layer = MaxMahalanobis(10, 2048, 1000)

        dot_values = remove_diagonal(dot_products(layer.centers))   # [N choose 2]
        angles = torch.acos(dot_values / 10 ** 2)   # [N choose 2]
        angles = radians_to_degrees(angles)   # [N choose 2]

        self.assert_tensor_almost_constant(angles, 1e-5)
        print(angles[0])

    def test_2048_1000_distances(self) -> None:
        layer = MaxMahalanobis(10, 2048, 1000)

        dist_values = remove_diagonal(euclidean_distances(layer.centers))   # [N choose 2]
        print(dist_values.min())
        print(dist_values.max())
