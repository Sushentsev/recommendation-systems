import numpy as np
import pandas as pd

from hw2.datasets.base import Dataset


class MembersDataset(Dataset):
    def _handle_bd(self):
        def category(value: int) -> str:
            if value == 0 or value > 100:
                return "<UNK>"
            elif 0 < value <= 16:
                return "child"
            elif 16 < value <= 30:
                return "young"
            elif 30 < value <= 45:
                return "middle_age"
            else:
                return "old_age"

        self._df["bd_category"] = self._df["bd"].apply(category).astype("category")
        self._df = self._df.drop(columns="bd")

    def _handle_gender(self):
        self._df["gender"] = self._df["gender"].cat.add_categories("<UNK>").fillna(value="<UNK>")

    def _handle_registration_init_time(self):
        self._df["registration_init_year"] = self._df["registration_init_time"].apply(lambda x: x.year).astype("int")
        self._df = self._df.drop(columns="registration_init_time")

    def _handle_expiration_date(self):
        self._df["expiration_date_year"] = self._df["expiration_date"].apply(lambda x: x.year).astype("int")
        self._df = self._df.drop(columns="expiration_date")

    def create_features(self, inplace: bool = False) -> "MembersDataset":
        if not inplace:
            dataset = MembersDataset(self._df).create_features(inplace=True)
            return dataset

        handlers = [self._handle_bd, self._handle_gender,
                    self._handle_registration_init_time,
                    self._handle_expiration_date]

        for handler in handlers:
            handler()

        return self

    @staticmethod
    def from_path(path: str) -> "MembersDataset":
        df = pd.read_csv(path, dtype={
            "msno": "category",
            "city": "category",
            "bd": np.uint8,
            "gender": "category",
            "registered_via": "category"}, parse_dates=["registration_init_time", "expiration_date"])

        return MembersDataset(df)
