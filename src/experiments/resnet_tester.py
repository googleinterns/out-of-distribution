import json
import os
from datetime import datetime
from typing import Optional, Any, List, Dict

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from root import from_root
from src.adversaries.adversary import Adversary
from src.adversaries.carlini_and_wagner import CarliniAndWagner, CarliniAndWagnerOutput
from src.adversaries.fgsm import FastGradientSignMethod
from src.adversaries.iterative_fgsm import IterativeFgsm
from src.adversaries.momentum_fgsm import MomentumFgsm
from src.adversaries.momentum_fgsm_transfer import MomentumFgsmTransfer
from src.adversaries.pgd import ProjectedGradientDescent
from src.adversaries.random_sampling import RandomSampling
from src.adversaries.spsa import SimultaneousPerturbationStochasticApproximation
from src.datasets.load_cifar10 import load_cifar10_infer
from src.datasets.load_mnist import load_mnist_infer
from src.datasets.load_svhn import load_svhn_infer
from src.experiments.config import create_resnet
from src.misc.collection_object import DictObject
from src.misc.test_tracker import Tracker
from src.misc.utils import set_deterministic_seed, model_device
from src.resnet.shared import ResNet_Gaussian, ResNet_Softmax, GaussianMode, SoftmaxMode


class Tester:
    model: nn.Module
    loader: DataLoader
    adversary: Optional[Adversary]
    out_name: str
    visualize_adversary: int

    def __init__(
        self, cfg_filepath: str, data_loader: DataLoader, adversary: Optional[Adversary], out_name: str,
        visualize_adversary: int
    ):
        set_deterministic_seed()
        with open(from_root(cfg_filepath), "r") as file:
            self.cfg = DictObject(json.load(file))

        os.makedirs(self.from_out("inference"), exist_ok=True)
        os.makedirs(self.from_out(f"inference_debug/{out_name}"), exist_ok=True)

        self.model = create_resnet(self.cfg)
        self.model = self.model.to(self.cfg.model.device)
        self.load_best_epoch()

        self.loader = data_loader
        self.adversary = adversary
        self.out_name = out_name
        self.visualize_adversary = visualize_adversary

        Tracker.reset()

    def load_best_epoch(self) -> None:
        with open(self.from_out("checkpoints/best_epoch.txt"), "r") as file:
            epoch = int(file.read())

        checkpoint_path = self.from_out(f"checkpoints/checkpoint_{epoch}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=model_device(self.model))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def from_out(self, relative_path: str) -> str:
        return os.path.join(from_root(self.cfg.out_dirpath), relative_path)

    def run(self) -> None:
        Tracker.n_batches = len(self.loader)
        Tracker.start_time = datetime.now()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        result = []
        for Tracker.batch, (images, labels) in enumerate(self.loader, 1):
            result += self.run_batch(images, labels)

        with open(self.from_out(f"inference/{self.out_name}.json"), "w") as file:
            json.dump(result, file)

    def run_batch(self, images: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, Any]]:
        images = images.to(model_device(self.model))
        labels = labels.to(model_device(self.model))

        if self.adversary is None:
            output = None
        else:
            output = self.adversary(self.model, images, labels)
            if Tracker.batch == 1:
                for i in range(min(images.size(0), self.visualize_adversary)):
                    debug_img = torch.stack([images[i], output.noises[i] + 0.5, output.perturbed_images[i]], dim=0)
                    filepath = self.from_out(f"inference_debug/{self.out_name}/image_{i + 1}.png")
                    torchvision.utils.save_image(debug_img, filepath, nrow=3)
            images = output.perturbed_images

        result = [{"true_label": label.item()} for label in labels]
        if isinstance(self.model, ResNet_Gaussian):
            gaussian_output = self.model(images, GaussianMode.BOTH)
            for im_result, im_log_probs, im_log_likelihoods in zip(
                result, gaussian_output.log_posteriors, gaussian_output.log_likelihoods
            ):
                im_result["log_probs"] = im_log_probs.tolist()
                im_result["log_likelihoods"] = im_log_likelihoods.tolist()
        else:
            assert isinstance(self.model, ResNet_Softmax)
            log_probs = self.model(images, SoftmaxMode.LOG_SOFTMAX)
            for im_result, im_log_probs in zip(result, log_probs):
                im_result["log_probs"] = im_log_probs.tolist()

        if isinstance(output, CarliniAndWagnerOutput):
            for im_result, noise_l2_norm in zip(result, output.l2_norm):
                im_result["noise_l2_norm"] = noise_l2_norm.item()

        Tracker.progress()
        return result


def main() -> None:
    run_id_datasets()
    run_ood_datasets()
    run_adversaries()
    run_spsa_adversary()
    run_momentum_adversary()


def run_id_datasets() -> None:
    for model_cfg_filepath in (
        "src/experiments/resnet_3stage/resnet20_softmax_cifar10.json",
        "src/experiments/resnet_3stage/resnet32_softmax_cifar10.json",
        "src/experiments/resnet_3stage/resnet44_softmax_cifar10.json",
        "src/experiments/resnet_3stage/resnet56_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet29v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet47v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet65v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet29v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet47v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet65v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_gaussian_cifar10.json"
    ):
        loader = load_cifar10_infer("train", 1024, 16)
        tester = Tester(model_cfg_filepath, loader, None, "id_train", -1)
        tester.run()

        loader = load_cifar10_infer("val", 1024, 16)
        tester = Tester(model_cfg_filepath, loader, None, "id_val", -1)
        tester.run()


def run_ood_datasets() -> None:
    for model_cfg_filepath in (
        "src/experiments/resnetv2_3stage/resnet29v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet47v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet65v2_gaussian_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_gaussian_cifar10.json"
    ):
        loader = load_svhn_infer("val", 1024, 16)
        tester = Tester(model_cfg_filepath, loader, None, "ood_svhn", -1)
        tester.run()

        loader = load_mnist_infer("val", 1024, 16)
        tester = Tester(model_cfg_filepath, loader, None, "ood_mnist", -1)
        tester.run()


def run_adversaries() -> None:
    adversaries = []
    out_names = []

    for epsilon in (1, 2, 4, 8):
        adversary = FastGradientSignMethod(epsilon / 255)
        adversaries.append(adversary)
        out_names.append(f"fgsm_{epsilon}")

        adversary = IterativeFgsm(epsilon / 255, 1 / 255)
        adversaries.append(adversary)
        out_names.append(f"iterative_fgsm_{epsilon}")

        adversary = MomentumFgsm(epsilon / 255, 10, 1.0)
        adversaries.append(adversary)
        out_names.append(f"momentum_fgsm_{epsilon}")

        adversary = ProjectedGradientDescent(epsilon / 255, 10, 2 / 255)
        adversaries.append(adversary)
        out_names.append(f"pgd_10_{epsilon}")

        adversary = ProjectedGradientDescent(epsilon / 255, 50, 2 / 255)
        adversaries.append(adversary)
        out_names.append(f"pgd_50_{epsilon}")

        adversary = ProjectedGradientDescent(epsilon / 255, 100, 2 / 255)
        adversaries.append(adversary)
        out_names.append(f"pgd_100_{epsilon}")

        adversary = ProjectedGradientDescent(epsilon / 255, 200, 2 / 255)
        adversaries.append(adversary)
        out_names.append(f"pgd_200_{epsilon}")

        adversary = RandomSampling(epsilon / 255, 200)
        adversaries.append(adversary)
        out_names.append(f"random_sampling_{epsilon}")

    adversary = CarliniAndWagner(9, 1000, 0.01, 0.005)
    adversaries.append(adversary)
    out_names.append(f"carlini_and_wagner")

    for model_cfg_filepath in (
        "src/experiments/resnet_3stage/resnet56_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_gaussian_cifar10.json"
    ):
        for adversary, out_name in zip(adversaries, out_names):
            loader = load_cifar10_infer("val", 1024, 16)
            tester = Tester(model_cfg_filepath, loader, adversary, out_name, 10)
            tester.run()


def run_spsa_adversary() -> None:
    adversaries = []
    out_names = []

    for epsilon in (1, 2, 4, 8):
        adversary = SimultaneousPerturbationStochasticApproximation(epsilon / 255, 0.01, 0.01, 128, 10)
        adversaries.append(adversary)
        out_names.append(f"spsa_10_{epsilon}")

    for model_cfg_filepath in (
        "src/experiments/resnet_3stage/resnet56_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_softmax_cifar10.json",
        "src/experiments/resnetv2_3stage/resnet83v2_gaussian_cifar10.json"
    ):
        for adversary, out_name in zip(adversaries, out_names):
            loader = load_cifar10_infer("val", 1, 16)
            tester = Tester(model_cfg_filepath, loader, adversary, out_name, 10)
            tester.run()


def run_momentum_adversary() -> None:
    adversaries = []
    out_names = []

    for epsilon in (1, 2, 4, 8):
        surrogate_cfg_filepath = "src/experiments/resnetv2_3stage/resnet83v2_softmax_cifar10.json"
        adversary = MomentumFgsmTransfer(surrogate_cfg_filepath, epsilon / 255, 10, 1.0)
        adversaries.append(adversary)
        out_names.append(f"momentum_fgsm_transfer_{epsilon}")

    model_cfg_filepath = "src/experiments/resnetv2_3stage/resnet83v2_gaussian_cifar10.json"
    for adversary, out_name in zip(adversaries, out_names):
        loader = load_cifar10_infer("val", 1024, 16)
        tester = Tester(model_cfg_filepath, loader, adversary, out_name, 10)
        tester.run()


if __name__ == "__main__":
    main()
