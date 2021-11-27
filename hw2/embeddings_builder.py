from typing import List, Union

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from hw2.datasets.train import TrainDataset


class EmbeddingsBuilder:
    def __init__(self, embedding_dim: int, random_state: int, verbose: bool = True):
        self._embedding_dim = embedding_dim
        self._random_state = random_state
        self._user_encoder = LabelEncoder()
        self._users = set()
        self._verbose = verbose

        self._word2vec = None
        self._user_embeddings = None

    def _fit_items(self, sessions_df: pd.DataFrame):
        if self._verbose:
            print("Fitting items")

        sessions = dict(sessions_df.groupby("msno").song_id.apply(list))
        sentences = [values for values in sessions.values() if len(values) > 0]
        self._word2vec = Word2Vec(vector_size=self._embedding_dim, window=5, min_count=5,
                                  seed=self._random_state)
        self._word2vec.build_vocab(sentences)
        self._word2vec.train(sentences, total_examples=self._word2vec.corpus_count, epochs=10)

    def _fit_users(self, sessions_df: pd.DataFrame):
        if self._verbose:
            print("Fitting users")

        positive_sessions = dict(sessions_df[sessions_df.target == 1].groupby("msno").song_id.apply(list))

        self._user_encoder.fit(list(positive_sessions.keys()))
        self._users = set(self._user_encoder.classes_)

        self._user_embeddings = np.zeros((len(self._users), self._embedding_dim))

        for user, user_positives in tqdm(positive_sessions.items(), total=len(positive_sessions),
                                         disable=not self._verbose):
            user_positives = [positive for positive in user_positives if self.has_item(positive)]
            if len(user_positives) > 0:
                user_encoded = self._user_encoder.transform([user])[0]
                self._user_embeddings[user_encoded] = self.get_item_embeddings(user_positives).mean(axis=0)

    def fit(self, dataset: TrainDataset) -> "EmbeddingsBuilder":
        self._fit_items(dataset.pandas_df)
        self._fit_users(dataset.pandas_df)
        return self

    def has_item(self, item: str) -> bool:
        return item in self._word2vec.wv

    def has_user(self, user: str) -> bool:
        return user in self._users

    def get_item_embeddings(self, items: Union[List[str], np.ndarray]) -> np.ndarray:
        return self._word2vec.wv[items]

    def get_user_embeddings(self, users: Union[List[str], np.ndarray]) -> np.ndarray:
        users_encoded = self._user_encoder.transform(users)
        return self._user_embeddings[users_encoded]

    @property
    def default_embedding(self) -> np.ndarray:
        return np.zeros(self._embedding_dim)
