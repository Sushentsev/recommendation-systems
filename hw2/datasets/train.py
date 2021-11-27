from typing import Generator, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from hw2.datasets.base import Dataset


class TrainDataset(Dataset):
    def reduce_by_members(self, size: int, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).reduce_by_members(size, inplace=True)
            return dataset

        self._df = self._df.groupby("msno").head(size).reset_index(drop=True)
        return self

    def remove_by_mask(self, mask, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).remove_by_mask(mask, inplace=True)
            return dataset

        self._df = self._df[~mask]
        return self

    def sort_by(self, column: str, inplace: bool = False) -> "TrainDataset":
        if not inplace:
            dataset = TrainDataset(self._df).sort_by(column, inplace=True)
            return dataset

        self._df = self._df.sort_values(by="msno")
        return self

    def group_split(self, n_splits: int) -> Generator:
        # Not used anymore. Split by query.
        group_kfold = GroupKFold(n_splits=n_splits)

        # df_sorted = self._df.sort_values(by="msno")
        data = self._df.drop("target", axis=1)
        groups = data.msno.cat.codes.to_numpy()

        for train_index, test_index in group_kfold.split(data, groups=groups):
            train_dataset = TrainDataset(self._df.iloc[train_index])
            test_dataset = TrainDataset(self._df.iloc[test_index])
            yield train_dataset, test_dataset

    def split(self, n_splits: int) -> Generator:
        splits = np.array_split(np.arange(len(self._df)), n_splits)

        for i in range(n_splits):
            mask = (np.arange(n_splits) != i)  # All except i
            train_index = np.hstack(splits[mask])
            test_index = splits[i]

            # Remove leaks.
            train_dataset = self._df.iloc[train_index]
            test_dataset = self._df.iloc[test_index]

            cols = ["msno", "song_id"]
            mask_2d = np.isin(test_dataset[cols], train_dataset[cols])
            mask = ~np.all(mask_2d, axis=1)

            yield TrainDataset(train_dataset), TrainDataset(test_dataset[mask])

    def add_features(self, name: str, values: Any):
        self._df[name] = values

    def drop_features(self, name: str):
        self._df = self._df.drop(columns=name)

    @property
    def queries(self) -> np.ndarray:
        return self._df.msno.cat.codes.to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self._df.target.to_numpy()

    @staticmethod
    def from_path(path: str) -> "TrainDataset":
        df = pd.read_csv(path, dtype={
            "msno": "category",
            "song_id": "category",
            "source_system_tab": "category",
            "source_screen_name": "category",
            "source_type": "category",
            "target": np.int})

        return TrainDataset(df)
