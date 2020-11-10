from src.misc.test_case import TestCase
from src.resnet.resnet_3stage_softmax import ResNet20_Softmax


class TestResNet20(TestCase):
    def test_parameters(self) -> None:
        model = ResNet20_Softmax(10)

        # the ResNet paper uses padded identity shortcuts instead of projection shortcuts to change dimensionality, so
        # don't include projection parameters in the count
        our_n_parameters = sum(param.numel() for name, param in model.named_parameters() if ".projection" not in name)
        self.assertTrue(265000 <= our_n_parameters < 275000)
