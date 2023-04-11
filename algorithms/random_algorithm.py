import random
from math import comb

import numpy as np

from algorithms.cb_algorithm import CB_Algorithm
from pdgraph import allocation_by_id


class Random_Allocation(CB_Algorithm):
    """
    Randomized colonel blotto allocation algorithm
    """

    def __init__(self, K: int, n: int):
        """
        Randomized allocation algorithm initializer
        :param K: the number of jobs (battlefields)
        :param n: discrete resources to use
        """
        super().__init__()

        self.K = K
        self.n = n

        self.num_decisions = comb(K + n - 1, K - 1)

    def generate_decision(self):
        """
        Generate an allocation decision
        :return: the allocation decision
        """
        return allocation_by_id(random.randint(0, self.num_decisions-1), self.K, self.n)

    def update(self, _: np.ndarray):
        pass
