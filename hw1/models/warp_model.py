import time
from collections import namedtuple
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from hw1.base import FactorizationModel
from hw1.utils import log_iter

Sample = namedtuple("Sample", "user pos_item neg_item queries n_neg_items")


class WARPModel(FactorizationModel):
    """
    Inspired by:
    1) https://building-babylon.net/2016/03/18/warp-loss-for-implicit-feedback-recommendation/
    2) https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/37180.pdf
    3) https://ethen8181.github.io/machine-learning/recsys/5_warp.html
    """

    def __init__(self, factors: int, lr: float, iterations: int, verbose: bool = False, verbose_every: int = 1):
        super().__init__(factors, iterations, verbose, verbose_every)
        self._lr = lr
        self._neg_tries = 1000
        self._triplet_acc = 0.
        self._correct_cnt = 0
        self._curr_queries = 0
        self._n_neg_items = 0
        self._queries = 0

    def _sample_negative(self, user_item: csr_matrix, user: int, pos_item: int) -> Optional[int]:
        def score(u: int, i: int) -> float:
            return self._U[u] @ self._I[i]

        def eval_margin(pos_score: float, neg_score: float) -> float:
            return 1 - pos_score + neg_score

        def is_correct(pos_score: float, neg_score: float) -> bool:
            return pos_score > neg_score

        self._curr_queries = 0
        n_users, n_items = user_item.shape
        pos_items = user_item[user].nonzero()[1]
        neg_items = np.setdiff1d(np.arange(n_items), pos_items)  # O(n)?
        self._n_neg_items = len(neg_items)

        if len(neg_items) > 0:
            neg_item = np.random.choice(neg_items)
            pos_score, neg_score = score(user, pos_item), score(user, neg_item)
            margin = eval_margin(pos_score, neg_score)

            self._curr_queries += 1
            self._correct_cnt += is_correct(pos_score, neg_score)

            while self._curr_queries < self._neg_tries and margin < 0:
                neg_item = np.random.choice(neg_items)
                neg_score = score(user, neg_item)
                margin = eval_margin(pos_score, neg_score)

                self._curr_queries += 1
                self._correct_cnt += is_correct(pos_score, neg_score)

            if margin > 0:
                return neg_item

    def _grad_step(self, user: int, pos_item: int, neg_item: int):
        rank_est = int(self._n_neg_items / self._curr_queries)
        weight = np.log(rank_est)

        self._U[user] -= self._lr * weight * (self._I[neg_item] - self._I[pos_item])
        self._I[pos_item] -= self._lr * weight * (-self._U[user])
        self._I[neg_item] -= self._lr * weight * self._U[user]

    def _grad_steps(self, user_item: csr_matrix):
        self._correct_cnt = self._queries = 0

        n_samples = user_item.count_nonzero()
        order = np.random.permutation(n_samples)
        users, items = user_item.nonzero()

        for user, pos_item in zip(users[order], items[order]):
            neg_item = self._sample_negative(user_item, user, pos_item)
            if neg_item is not None:
                self._grad_step(user, pos_item, neg_item)

            self._queries += self._curr_queries

        self._triplet_acc = self._correct_cnt / self._queries

    def fit(self, user_item: csr_matrix) -> "WARPModel":
        self._start_time = time.time()
        self.init_matrices(*user_item.shape)

        for iteration in tqdm(range(self._iterations), disable=not self._verbose):
            self._grad_steps(user_item)

            if self._verbose and (iteration + 1) % self._verbose_every == 0:
                log_iter(iteration + 1, {"Triplet acc": self._triplet_acc}, time.time() - self._start_time)

        return self
