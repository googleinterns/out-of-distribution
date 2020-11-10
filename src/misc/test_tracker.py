from datetime import datetime


class Tracker:
    batch: int
    n_batches: int
    start_time: datetime

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls} should not be instantiated!")

    @staticmethod
    def reset() -> None:
        Tracker.iteration = 1
        Tracker.start_time = datetime.now()

    @staticmethod
    def progress() -> None:
        batch = f"Batch {Tracker.batch} of {Tracker.n_batches}"
        elapsed_time = f"Elapsed time {datetime.now() - Tracker.start_time}"

        message = f"Testing | {batch} | {elapsed_time}"
        print(message)
