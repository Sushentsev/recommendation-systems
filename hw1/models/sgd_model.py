import time
from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from hw1.base import FactorizationModel
from hw1.utils import rmse, log_iter


class SGDModel(FactorizationModel):
    def __init__(self, factors: int, lr: float, iterations: int, verbose: bool = False, verbose_every: int = 1):
        super().__init__(factors, iterations, verbose, verbose_every)
        self._lr = lr

    def _grad_steps(self, samples: List[Tuple[int, int, int]]):
        shuffled_samples = np.random.permutation(samples)
        for u, i, v in shuffled_samples:
            error = self._U[u] @ self._I[i] - v
            self._U[u] -= self._lr * error * self._I[i]
            self._I[i] -= self._lr * error * self._U[u]

    def fit(self, user_item: csr_matrix) -> "SGDModel":
        self._start_time = time.time()
        self.init_matrices(*user_item.shape)

        rows, cols = user_item.nonzero()
        samples = [(i, j, v) for i, j, v in zip(rows, cols, user_item.data)]

        for iteration in tqdm(range(self._iterations), disable=not self._verbose):
            self._grad_steps(samples)

            if self._verbose and (iteration + 1) % self._verbose_every == 0:
                y_pred = (self._U @ self._I.T)[user_item.nonzero()]
                y_true = user_item.data
                log_iter(iteration + 1, {"RMSE": rmse(y_pred, y_true)}, time.time() - self._start_time)

        return self
