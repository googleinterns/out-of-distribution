import os
import ssl

import torch
import torchvision

from root import from_root
from src.datasets.load_imagenet import IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD

OUT_FILEPATH = from_root("weights/softmax_resnet50_imagenet/pretrained.pth")


def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    torch_model = torchvision.models.resnet50(pretrained=True)

    new_state_dict = {
        "normalize.mean": torch.tensor(IMAGENET_TRAIN_MEAN).reshape(1, -1, 1, 1),
        "normalize.std": torch.tensor(IMAGENET_TRAIN_STD).reshape(1, -1, 1, 1)
    }

    for key, value in torch_model.state_dict().items():
        breadcrumbs = key.split(".")

        # change "layer1" to "stage2"
        if breadcrumbs[0].startswith("layer"):
            index = int(breadcrumbs[0][5:])
            breadcrumbs[0] = "stage" + str(index + 1)

        # change "downsample" to "projection"
        if len(breadcrumbs) >= 3 and breadcrumbs[2] == "downsample":
            breadcrumbs[2] = "projection"

        new_key = ".".join(breadcrumbs)
        new_state_dict[new_key] = value

    os.makedirs(os.path.dirname(OUT_FILEPATH), exist_ok=True)
    torch.save({"model_state_dict": new_state_dict}, OUT_FILEPATH)


if __name__ == "__main__":
    main()
