import os

import pandas as pd

from hw2.load_data import load_song_info, load_songs


def handle_artist_name(features_df: pd.DataFrame):
    def artist_count(value: str):
        if value == "<UNK>":
            return 0

        return value.count("|") + value.count("and") + value.count(",") + value.count("feat") + value.count("&") + 1

    features_df.artist_name = features_df.artist_name.cat.add_categories("<UNK>").fillna(value="<UNK>")
    features_df["artist_name_count"] = features_df.artist_name.apply(artist_count).astype("int")


def handle_composer(features_df: pd.DataFrame):
    def composer_count(value: str):
        if value == "<UNK>":
            return 0

        return sum(map(value.count, ["|", "/", "\\", ";"])) + 1

    features_df.composer = features_df.composer.cat.add_categories("<UNK>").fillna(value="<UNK>")
    features_df["composer_count"] = features_df.composer.apply(composer_count).astype("int")


def handle_lyricist(features_df: pd.DataFrame):
    def lyricist_count(value: str):
        if value == "<UNK>":
            return 0
        return sum(map(value.count, ["|", "/", "\\", ";"])) + 1

    features_df.lyricist = features_df.lyricist.cat.add_categories("<UNK>").fillna(value="<UNK>")
    features_df["lyricists_count"] = features_df.lyricist.apply(lyricist_count).astype("int")


def handle_language(features_df: pd.DataFrame):
    features_df.language = features_df.language.fillna(value="-1.0")


def handle_genre_ids(features_df: pd.DataFrame):
    def genres_count(value: str):
        if value == "<UNK>":
            return 0
        return sum(map(value.count, ["|"])) + 1

    features_df.genre_ids = features_df.genre_ids.cat.add_categories("<UNK>").fillna(value="<UNK>")
    features_df["genres_count"] = features_df.lyricist.apply(genres_count).astype("int")


def handle_name(features_df: pd.DataFrame):
    features_df.drop(columns="name", inplace=True)


def handle_isrc(features_df: pd.DataFrame):
    def isrc_to_year(isrc) -> int:
        if type(isrc) == str:
            if int(isrc[5:7]) > 17:
                return 1900 + int(isrc[5:7])
            else:
                return 2000 + int(isrc[5:7])
        else:
            return -1

    features_df["isrc_year"] = features_df.isrc.apply(isrc_to_year).astype("category")
    features_df.drop(columns="isrc", inplace=True)


def create_song_features(data_dir: str):
    songs_df = load_songs(os.path.join(data_dir, "songs.csv"))
    song_info_df = load_song_info(os.path.join(data_dir, "song_extra_info.csv"))
    songs_df = songs_df.merge(song_info_df, on="song_id", how="left")

    songs_features_df = songs_df

    handle_artist_name(songs_features_df)
    handle_composer(songs_features_df)
    handle_lyricist(songs_features_df)
    handle_language(songs_features_df)
    handle_name(songs_features_df)
    handle_isrc(songs_features_df)
    handle_genre_ids(songs_features_df)

    return songs_features_df
