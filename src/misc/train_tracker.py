from datetime import datetime
from enum import Enum
from math import inf
from typing import Union


class Mode(Enum):
    TRAINING = 0
    INFER_TRAIN = 1
    INFER_VAL = 2


class Tracker:
    epoch: int
    n_epochs: Union[int, float]
    batch: int
    n_batches: int
    global_batch: int
    mode: Mode
    start_time: datetime
    best_val_accuracy: float

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls} should not be instantiated!")

    @staticmethod
    def reset(n_epochs: Union[int, float]) -> None:
        Tracker.n_epochs = n_epochs
        Tracker.epoch = 1
        Tracker.global_batch = 1
        Tracker.start_time = datetime.now()
        Tracker.best_val_accuracy = -inf

    @staticmethod
    def progress() -> None:
        mode = {
            Mode.TRAINING: "Training",
            Mode.INFER_TRAIN: "Inferring train set",
            Mode.INFER_VAL: "Inferring validation set"
        }[Tracker.mode]

        batch = f"Batch {Tracker.batch} of {Tracker.n_batches}"
        elapsed_time = f"Elapsed time {datetime.now() - Tracker.start_time}"

        if Tracker.mode == Mode.TRAINING:
            epoch = f"Epoch {Tracker.epoch} of {Tracker.n_epochs}"
            global_batch = f"Global batch {Tracker.global_batch}"
            message = f"{mode} | {epoch} | {batch} | {global_batch} | {elapsed_time}"
        else:
            epoch = f"Epoch {Tracker.epoch} of {Tracker.n_epochs}"
            message = f"{mode} | {epoch} | {batch} | {elapsed_time}"

        print(message)
