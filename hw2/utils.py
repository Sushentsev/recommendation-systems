import pickle
from typing import Any


def save_pickle(data: Any, path: str):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)
