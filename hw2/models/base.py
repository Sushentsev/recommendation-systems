from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from hw2.datasets.train import TrainDataset
from hw2.metrics import ndcg, auc_per_query


class Model(ABC):
    def __init__(self, random_state: int, verbose: bool = True):
        self._random_state = random_state
        self._verbose = verbose

    @abstractmethod
    def fit(self, dataset: TrainDataset) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: TrainDataset) -> np.ndarray:
        raise NotImplementedError

    def cv_scores(self, dataset: TrainDataset, n_splits: int) -> Dict[str, List[float]]:
        metrics = {"NDCG": [], "ROC_AUC": []}

        for train_dataset, test_dataset in dataset.split(n_splits, self._random_state):
            if self._verbose:
                print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

            self.fit(train_dataset)
            scores = self.predict(test_dataset)

            metrics["NDCG"].append(ndcg(test_dataset.queries, scores, test_dataset.labels))
            metrics["ROC_AUC"].append(auc_per_query(test_dataset.queries, scores, test_dataset.labels))

            if self._verbose:
                print(f"NDCG: {metrics['NDCG'][-1]:.4f} | ROC AUC: {metrics['ROC_AUC'][-1]:.4f}")

        return metrics
