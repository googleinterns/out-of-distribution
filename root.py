import os

ROOT = os.path.abspath(os.path.dirname(__file__))


def from_root(relative_path: str) -> str:
    return os.path.join(ROOT, relative_path)
