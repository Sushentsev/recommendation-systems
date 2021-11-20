from typing import List
import numpy as np

from hw2.embeddings import EmbeddingsBuilder


class EmbeddingBasedModel:
    def __init__(self, embeddings: EmbeddingsBuilder):
        self._embeddings = embeddings

    def _get_user_emb(self, user: str) -> np.ndarray:
        if not self._embeddings.has_user(user):
            return self._embeddings.default_embedding

        return self._embeddings.get_user_embeddings([user])[0]

    def _get_item_emb(self, item: str) -> np.ndarray:
        if not self._embeddings.has_item(item):
            return self._embeddings.default_embedding

        return self._embeddings.get_item_embeddings([item])[0]

    def rank_scores(self, user: str, items: List[str]) -> np.ndarray:
        user_emb = self._get_user_emb(user)
        item_embs = np.vstack([self._get_item_emb(item) for item in items])
        scores = item_embs @ user_emb
        return scores
