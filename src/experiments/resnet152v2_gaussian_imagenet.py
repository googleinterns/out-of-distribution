from src.experiments.resnet_trainer import Trainer


def main() -> None:
    trainer = Trainer("src/experiments/resnetv2_4stage/resnet152v2_gaussian_imagenet.json")
    trainer.run()


if __name__ == "__main__":
    main()
