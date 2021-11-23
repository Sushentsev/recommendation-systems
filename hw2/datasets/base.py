from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import pandas as pd


class Dataset(ABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def merge(self, dataset: "Dataset", on: str, how: str) -> "Dataset":
        self._df = self._df.merge(dataset.pandas_df, on=on, how=how)
        return self

    def to_category(self, columns: List[str]):
        for column in columns:
            self._df[column] = self._df[column].astype("category")

    def fill_na_category(self, columns: List[str]):
        for column in columns:
            self._df[column] = self._df[column].cat.add_categories("<UNK>").fillna(value="<UNK>")

    @property
    def pandas_df(self, indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        if indices is not None:
            return self._df.iloc[indices]

        return self._df

    def __len__(self) -> int:
        return len(self._df)

    @staticmethod
    @abstractmethod
    def from_path(path: str) -> "Dataset":
        raise NotImplementedError
