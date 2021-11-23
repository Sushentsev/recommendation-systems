from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from hw2.datasets.train import TrainDataset
from hw2.metrics import ndcg, auc_per_query


class Model(ABC):
    @abstractmethod
    def fit(self, dataset: TrainDataset) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: TrainDataset) -> np.ndarray:
        raise NotImplementedError

    def cv_scores(self, dataset: TrainDataset, n_splits: int) -> Dict[str, List[float]]:
        metrics = {"NDCG": [], "ROC_AUC": []}

        for train_dataset, test_dataset in dataset.split(n_splits):
            print(f"Train dataset size: {len(train_dataset)} | Test dataset size: {len(test_dataset)}")

            self.fit(train_dataset)
            scores = self.predict(test_dataset)

            metrics["NDCG"].append(ndcg(test_dataset.queries, scores, test_dataset.labels))
            metrics["ROC_AUC"].append(auc_per_query(test_dataset.queries, scores, test_dataset.labels))

            print(f"NDCG: {metrics['NDCG'][-1]} | ROC AUC {metrics['ROC_AUC'][-1]}")

        return metrics