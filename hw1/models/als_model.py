import time

import numpy as np
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm

from hw1.base import FactorizationModel
from hw1.utils import log_iter, rmse


class ALSModel(FactorizationModel):
    def __init__(self, factors: int, iterations: int, lambd: float = 0., verbose: bool = False, verbose_every: int = 1):
        super().__init__(factors, iterations, verbose, verbose_every)
        self._lambd = lambd

    def _grad_steps(self, user_item: csr_matrix, X: np.ndarray, Y: np.ndarray):
        m1 = np.linalg.inv(Y.T @ Y + self._lambd * np.identity(self._factors))
        m2 = csr_matrix(m1 @ Y.T)
        for index in range(X.shape[0]):
            X[index] = (m2 @ user_item.getrow(index).T).toarray().squeeze()

    def fit(self, user_item: csr_matrix) -> "ALSModel":
        self._start_time = time.time()
        self.init_matrices(*user_item.shape)

        for iteration in tqdm(range(self._iterations), disable=not self._verbose):
            self._grad_steps(user_item, self._U, self._I)
            self._grad_steps(user_item.T, self._I, self._U)

            if self._verbose and (iteration + 1) % self._verbose_every == 0:
                y_pred = (self._U @ self._I.T)[user_item.nonzero()]
                y_true = user_item.data
                log_iter(iteration + 1, {"RMSE": rmse(y_pred, y_true)}, time.time() - self._start_time)

        return self
