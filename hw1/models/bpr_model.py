import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit
from tqdm import tqdm

from hw1.base import FactorizationModel
from hw1.utils import log_iter


class BPRModel(FactorizationModel):
    def __init__(self, factors: int, lr: float, iterations: int, lambd: float = 0.,
                 verbose: bool = False, verbose_every: int = 1):
        super().__init__(factors, iterations, verbose, verbose_every)
        self._lr = lr
        self._lambd = lambd
        self._correct_cnt = 0
        self._triplet_acc = 0.

    @staticmethod
    def _sample_negative(user_item: csr_matrix, user: int) -> int:
        neg_item = np.random.choice(user_item.shape[1])
        while user_item[user, neg_item] != 0:
            neg_item = np.random.choice(user_item.shape[1])
        return neg_item

    def _grad_step(self, user: int, pos_item: int, neg_item: int):
        score = expit(self._U[user] @ (self._I[neg_item] - self._I[pos_item]))
        self._correct_cnt += score < 0.5

        grad_user = score * (self._I[neg_item] - self._I[pos_item]) + self._lambd * self._U[user]
        grad_pos = score * -self._U[user] + self._lambd * self._I[pos_item]
        grad_neg = score * self._U[user] + self._lambd * self._I[neg_item]

        self._U[user] -= self._lr * grad_user
        self._I[pos_item] -= self._lr * grad_pos
        self._I[neg_item] -= self._lr * grad_neg

    def _grad_steps(self, user_item: csr_matrix):
        self._triplet_acc = self._correct_cnt = 0
        n_samples = user_item.count_nonzero()
        order = np.random.permutation(n_samples)
        users, items = user_item.nonzero()

        for user, pos_item in zip(users[order], items[order]):
            neg_item = self._sample_negative(user_item, user)
            self._grad_step(user, pos_item, neg_item)

        self._triplet_acc = self._correct_cnt / n_samples

    def fit(self, user_item: csr_matrix) -> "BPRModel":
        self._start_time = time.time()
        self.init_matrices(*user_item.shape)

        for iteration in tqdm(range(self._iterations), disable=not self._verbose):
            self._grad_steps(user_item)

            if self._verbose and (iteration + 1) % self._verbose_every == 0:
                log_iter(iteration + 1, {"Triplet acc": self._triplet_acc}, time.time() - self._start_time)

        return self
