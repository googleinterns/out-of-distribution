import torchvision

from src.misc.test_case import TestCase
from src.resnet.resnet_4stage_backbone import ResNet50_Backbone


class TestResNet50(TestCase):
    def test_parameters(self) -> None:
        torch_model = torchvision.models.resnet50(pretrained=False)
        torch_n_parameters = sum(param.numel() for name, param in torch_model.named_parameters() if "fc" not in name)

        our_model = ResNet50_Backbone()
        our_n_parameters = sum(param.numel() for param in our_model.parameters())
        self.assertEqual(torch_n_parameters, our_n_parameters)
