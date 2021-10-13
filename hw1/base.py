from abc import abstractmethod, ABC
from typing import Tuple

import numpy as np
from scipy.sparse.csr import csr_matrix


class FactorizationModel(ABC):
    def __init__(self, factors: int, iterations: int, verbose: bool = False, verbose_every: int = 1):
        self._factors = factors
        self._iterations = iterations
        self._verbose = verbose
        self._verbose_every = verbose_every
        self._U = self._I = None
        self._start_time = None

    def init_matrices(self, n_users: int, n_items: int):
        self._U = np.random.uniform(0, np.sqrt(1 / self._factors), size=(n_users, self._factors))
        self._I = np.random.uniform(0, np.sqrt(1 / self._factors), size=(n_items, self._factors))

    @abstractmethod
    def fit(self, user_item: csr_matrix) -> "FactorizationModel":
        raise NotImplementedError

    def get_user_emb(self, user_id: int) -> np.ndarray:
        return self._U[user_id]

    def get_item_emb(self, item_id: int) -> np.ndarray:
        return self._I[item_id]

    def factorize(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._U, self._I
