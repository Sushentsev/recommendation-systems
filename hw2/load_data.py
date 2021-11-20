import numpy as np
import pandas as pd


def load_members(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={
        "msno": "category",
        "city": "category",
        "bd": np.uint8,
        "gender": "category",
        "registered_via": "category"}, parse_dates=["registration_init_time", "expiration_date"])


def load_song_info(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={
        "song_id": "category",
        "name": "category"})


def load_songs(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={
        "song_id": "category",
        "song_length": np.int32,
        "genre_ids": "category",
        "artist_name": "category",
        "composer": "category",
        "lyricist": "category",
        "language": "category"})


def load_train(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={
        "msno": "category",
        "song_id": "category",
        "source_system_tab": "category",
        "source_screen_name": "category",
        "source_type": "category",
        "target": np.uint8})
