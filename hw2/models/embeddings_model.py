import numpy as np

from hw2.datasets.train import TrainDataset
from hw2.embeddings_builder import EmbeddingsBuilder
from hw2.models.base import Model


class EmbeddingModel(Model):
    def __init__(self, embeddings: EmbeddingsBuilder):
        self._embeddings = embeddings

    def fit(self, dataset: TrainDataset) -> "EmbeddingModel":
        return self

    def predict(self, dataset: TrainDataset) -> np.ndarray:
        scores = []

        for user, item in dataset.pandas_df[["msno", "song_id"]].to_numpy():
            user_emb = self._get_user_emb(user)
            item_emb = self._get_item_emb(item)
            scores.append(user_emb @ item_emb)

        return np.array(scores)

    def _get_user_emb(self, user: str) -> np.ndarray:
        if not self._embeddings.has_user(user):
            return self._embeddings.default_embedding

        return self._embeddings.get_user_embeddings([user])[0]

    def _get_item_emb(self, item: str) -> np.ndarray:
        if not self._embeddings.has_item(item):
            return self._embeddings.default_embedding

        return self._embeddings.get_item_embeddings([item])[0]
