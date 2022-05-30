import numpy as np

from hw2.datasets.train import TrainDataset
from hw2.embeddings_builder import EmbeddingsBuilder
from hw2.models.base import Model


class EmbeddingModel(Model):
    def __init__(self, embedding_dim: int, random_state: int, verbose: bool = True):
        super().__init__(random_state, verbose)
        self._embeddings = EmbeddingsBuilder(embedding_dim, random_state)

    def fit(self, dataset: TrainDataset) -> "EmbeddingModel":
        self._embeddings.fit(dataset)
        return self

    def predict(self, dataset: TrainDataset) -> np.ndarray:
        scores = np.zeros(len(dataset))

        users = dataset.pandas_df["msno"].to_numpy()
        items = dataset.pandas_df["song_id"].to_numpy()

        mask = np.array([self._embeddings.has_user(user) and self._embeddings.has_item(item)
                         for user, item in zip(users, items)])

        user_embs = self._embeddings.get_user_embeddings(users[mask])
        item_embs = self._embeddings.get_item_embeddings(items[mask])
        scores[mask] = np.sum(user_embs * item_embs, axis=1)

        return scores

    # def _get_user_emb(self, user: str) -> np.ndarray:
    #     if not self._embeddings.has_user(user):
    #         return self._embeddings.default_embedding
    #
    #     return self._embeddings.get_user_embeddings([user])[0]
    #
    # def _get_item_emb(self, item: str) -> np.ndarray:
    #     if not self._embeddings.has_item(item):
    #         return self._embeddings.default_embedding
    #
    #     return self._embeddings.get_item_embeddings([item])[0]
