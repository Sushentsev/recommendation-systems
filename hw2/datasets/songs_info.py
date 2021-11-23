import pandas as pd

from hw2.datasets.base import Dataset


class SongsInfoDataset(Dataset):
    @staticmethod
    def from_path(path: str) -> "SongsInfoDataset":
        df = pd.read_csv(path, dtype={
            "song_id": "category",
            "name": "category"})

        return SongsInfoDataset(df)
