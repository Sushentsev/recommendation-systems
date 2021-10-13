import time
from collections import namedtuple
from typing import Optional, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from tqdm import tqdm

from hw1.base import FactorizationModel
from hw1.utils import log_iter

Sample = namedtuple("Sample", "user pos_item neg_item")


class BPRModel(FactorizationModel):
    def __init__(self, factors: int, lr: float, iterations: int, lambd: float = 0.,
                 verbose: bool = False, verbose_every: int = 1):
        super().__init__(factors, iterations, verbose, verbose_every)
        self._lr = lr
        self._lambd = lambd
        self._is_sampled = None
        self._triplet_acc = 0.

    def _sample_triplet(self, user_item: csr_matrix) -> Optional[Sample]:
        n_users, n_items = user_item.shape
        user = np.random.choice(np.arange(n_users)[~self._is_sampled])
        self._is_sampled[user] = True
        pos_items = user_item[user].nonzero()[1]

        if len(pos_items) > 0:
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)  # Fine for sparse matrix

            return Sample(user, pos_item, neg_item)

    def _grad_steps(self, user_item: csr_matrix):
        correct_cnt = 0
        n_users = user_item.shape[0]
        self._is_sampled = np.array([False] * n_users)
        for _ in range(n_users):
            sample = self._sample_triplet(user_item)
            if sample is not None:
                score = expit(self._U[sample.user] @ (self._I[sample.neg_item] - self._I[sample.pos_item]))
                correct_cnt += score < 0.5

                grad_user = score * (self._I[sample.neg_item] - self._I[sample.pos_item]) + self._lambd * self._U[
                    sample.user]
                grad_pos = score * -self._U[sample.user] + self._lambd * self._I[sample.pos_item]
                grad_neg = score * self._U[sample.user] + self._lambd * self._I[sample.neg_item]
                self._U[sample.user] -= self._lr * grad_user
                self._I[sample.pos_item] -= self._lr * grad_pos
                self._I[sample.neg_item] -= self._lr * grad_neg

        self._triplet_acc = correct_cnt / n_users

    def fit(self, user_item: csr_matrix) -> "BPRModel":
        self._start_time = time.time()
        self.init_matrices(*user_item.shape)

        for iteration in tqdm(range(self._iterations), disable=not self._verbose):
            self._grad_steps(user_item)

            if self._verbose and (iteration + 1) % self._verbose_every == 0:
                log_iter(iteration + 1, {"Triplet acc": self._triplet_acc}, time.time() - self._start_time)

        return self
