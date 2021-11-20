from typing import List, Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class EmbeddingsBuilder:
    def __init__(self, embedding_dim: int, random_state: Optional[int] = None):
        self._embedding_dim = embedding_dim
        self._random_state = random_state
        self._user_encoder = LabelEncoder()
        self._users = set()

        self._item_embeddings = None
        self._user_embeddings = None

        self._default_embedding = np.zeros(embedding_dim)

    def _fit_items(self, sessions_df: pd.DataFrame):
        sessions = dict(sessions_df.groupby("msno").song_id.apply(list))
        sentences = [values for values in sessions.values() if len(values) > 0]
        self._item_embeddings = Word2Vec(sentences=sentences, vector_size=self._embedding_dim,
                                         window=5, min_count=5, seed=self._random_state)

    def _fit_users(self, sessions_df: pd.DataFrame):
        positive_sessions = dict(sessions_df[sessions_df.target == 1].groupby("msno").song_id.apply(list))
        self._user_encoder.fit(list(positive_sessions.keys()))
        self._users = set(self._user_encoder.classes_)
        n_users = len(self._users)
        self._user_embeddings = np.zeros((n_users, self._embedding_dim))

        for user, user_positives in tqdm(positive_sessions.items(), "Fitting users"):
            user_positives = [positive for positive in user_positives if self.has_item(positive)]
            if len(user_positives) > 0:
                user_encoded = self._user_encoder.transform([user])[0]
                self._user_embeddings[user_encoded] = self.get_item_embeddings(user_positives).mean(axis=0)

    def fit(self, sessions_df: pd.DataFrame) -> "EmbeddingsBuilder":
        self._fit_items(sessions_df)
        self._fit_users(sessions_df)
        return self

    def has_item(self, item: str) -> bool:
        return item in self._item_embeddings.wv

    def has_user(self, user: str) -> bool:
        return user in self._users

    def get_item_embeddings(self, items: List[str]) -> np.ndarray:
        return self._item_embeddings.wv[items]

    def get_user_embeddings(self, users: List[str]) -> np.ndarray:
        users_encoded = self._user_encoder.transform(users)
        return self._user_embeddings[users_encoded]

    @property
    def default_embedding(self) -> np.ndarray:
        return self._default_embedding
