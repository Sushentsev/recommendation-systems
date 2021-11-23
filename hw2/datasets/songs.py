import numpy as np
import pandas as pd

from hw2.datasets.base import Dataset
from hw2.datasets.songs_info import SongsInfoDataset


class SongsDataset(Dataset):
    def _handle_genre_ids(self):
        def counts(value: str):
            if value == "<UNK>":
                return 0
            return sum(map(value.count, ["|"])) + 1

        self._df["genre_ids"] = self._df["genre_ids"].cat.add_categories("<UNK>").fillna(value="<UNK>")
        self._df["genres_count"] = self._df["genre_ids"].apply(counts).astype("int")

    def _handle_artist_name(self):
        def counts(value: str):
            if value == "<UNK>":
                return 0
            return value.count("|") + value.count("and") + value.count(",") + value.count("feat") + value.count("&") + 1

        self._df["artist_name"] = self._df["artist_name"].cat.add_categories("<UNK>").fillna(value="<UNK>")
        self._df["artist_name_count"] = self._df["artist_name"].apply(counts).astype("int")

    def _handle_composer(self):
        def counts(value: str):
            if value == "<UNK>":
                return 0
            return sum(map(value.count, ["|", "/", "\\", ";"])) + 1

        self._df["composer"] = self._df["composer"].cat.add_categories("<UNK>").fillna(value="<UNK>")
        self._df["composer_count"] = self._df["composer"].apply(counts).astype("int")

    def _handle_lyricist(self):
        def counts(value: str):
            if value == "<UNK>":
                return 0
            return sum(map(value.count, ["|", "/", "\\", ";"])) + 1

        self._df["lyricist"] = self._df["lyricist"].cat.add_categories("<UNK>").fillna(value="<UNK>")
        self._df["lyricists_count"] = self._df["lyricist"].apply(counts).astype("int")

    def _handle_language(self):
        self._df["language"] = self._df["language"].fillna(value="-1.0")

    def _handle_name(self):
        self._df = self._df.drop(columns="name")

    def _handle_isrc(self):
        def to_year(isrc) -> int:
            if type(isrc) == str:
                if int(isrc[5:7]) > 17:
                    return 1900 + int(isrc[5:7])
                else:
                    return 2000 + int(isrc[5:7])
            else:
                return -1

        self._df["isrc_year"] = self._df["isrc"].apply(to_year).astype("category")
        self._df = self._df.drop(columns="isrc")

    def create_features(self, songs_info_dataset: SongsInfoDataset, inplace: bool = False) -> "SongsDataset":
        if not inplace:
            dataset = SongsDataset(self._df).create_features(songs_info_dataset, inplace=True)
            return dataset

        self.merge(songs_info_dataset, on="song_id", how="left")

        handlers = [self._handle_genre_ids, self._handle_artist_name, self._handle_composer,
                    self._handle_lyricist, self._handle_language, self._handle_name, self._handle_isrc]

        for handler in handlers:
            handler()

        return self

    @staticmethod
    def from_path(path: str) -> "SongsDataset":
        df = pd.read_csv(path, dtype={
            "song_id": "category",
            "song_length": np.int32,
            "genre_ids": "category",
            "artist_name": "category",
            "composer": "category",
            "lyricist": "category",
            "language": "category"})

        return SongsDataset(df)
