import numpy as np
from catboost import CatBoostRanker, Pool

from hw2.datasets.train import TrainDataset
from hw2.models.base import Model


class CatBoostModel(Model):
    def __init__(self, loss_function: str, iterations: int, task_type: str, random_state: int, verbose: bool = True):
        super().__init__(verbose)
        self._model = CatBoostRanker(loss_function=loss_function, iterations=iterations,
                                     task_type=task_type, random_state=random_state, verbose=verbose)

    def fit(self, dataset: TrainDataset) -> "CatBoostModel":
        pool = CatBoostModel.to_pool(dataset)
        self._model.fit(pool)
        return self

    def predict(self, dataset: TrainDataset) -> np.ndarray:
        pool = CatBoostModel.to_pool(dataset)
        pred = self._model.predict(pool)
        return pred

    @staticmethod
    def to_pool(dataset: TrainDataset) -> Pool:
        cat_features = dataset.pandas_df.select_dtypes(include=["category"]).columns.to_numpy()

        data = dataset.pandas_df.drop("target", axis=1)
        label = dataset.pandas_df.target.to_numpy()
        group_id = dataset.pandas_df.msno.cat.codes.to_numpy()

        pool = Pool(data=data, label=label, group_id=group_id,
                    cat_features=cat_features, has_header=True)

        return pool
