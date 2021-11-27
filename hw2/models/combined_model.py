from hw2.datasets.train import TrainDataset
from hw2.models.base import Model
from hw2.models.catboost_model import CatBoostModel
from hw2.models.embeddings_model import EmbeddingModel
import numpy as np

class CombinedModel(Model):
    def __init__(self, loss_function: str, iterations: int, embedding_dim: int,
                 task_type: str, random_state: int, verbose: bool = True):
        self._catboost_model = CatBoostModel(loss_function, iterations, task_type, random_state, verbose)
        self._embedding_model = EmbeddingModel(embedding_dim, random_state, verbose)

    def _add_scores(self, dataset: TrainDataset):
        scores = self._embedding_model.predict(dataset)
        dataset.add_features("emb_score", scores)

    def _drop_scores(self, dataset: TrainDataset):
        dataset.drop_features("emb_score")

    def fit(self, dataset: TrainDataset) -> "CombinedModel":
        self._embedding_model.fit(dataset)
        self._add_scores(dataset)
        self._catboost_model.fit(dataset)
        self._drop_scores(dataset)
        return self

    def predict(self, dataset: TrainDataset) -> np.ndarray:
        self._add_scores(dataset)
        pred = self._catboost_model.predict(dataset)
        self._drop_scores(dataset)
        return pred
