from math import exp, sqrt, log
from random import choice

import numpy as np

from algorithms.cb_algorithm import CB_Algorithm
from pdgraph import make_discrete_allocation


class MARA(CB_Algorithm):
    """
    Resource-allocation algorithm for the multi-armed problem from Dagan and Crammer. (https://proceedings.mlr.press/v83/dagan18a)
    """

    def __init__(self, K: int, m: int = None, c: float = 2.5):
        """
        Multi-Armed Resource-Allocation algorithm initializer
        :param K: the number of jobs (battlefields)
        :param m: discrete resources to use (None by default; use continuous allocation)
        :param c: arbitrary coefficient >2 for computing allocation offset r (2.5 by default)
        """
        super().__init__()

        if c <= 2:
            raise ValueError('c must be greater than 2')
        self.c = c
        self.K = K
        self.m = m

        self.v_prob = np.zeros(shape=(1, K))
        self.v_det = np.zeros(shape=(1, K))

        self.s_det = np.empty(shape=(0, K))
        self.r = np.empty(shape=(0, K))
        self.M = np.empty(shape=(0, K))
        self.X = np.empty(shape=(0, K), dtype=np.bool_)

        self.t = 1

    def generate_decision(self):
        """
        Generate an allocation decision
        :return: the allocation decision
        """
        resource = 1

        r_t = np.empty(shape=self.K)
        M_t = np.empty(shape=self.K)

        v_max = np.maximum(self.v_det[-1], self.v_prob[-1])

        for k in np.argsort(v_max):
            if self.v_det[-1, k] > 0:
                r_t[k] = self.c * self.v_det[-1, k] * exp(-self.s_det[-1, k] / (self.c * self.v_det[-1, k]))
            else:
                r_t[k] = 0

            if self.v_det[-1, k] == 0:
                M_t[k] = 1 / (self.K * (2 ** (self.t - 1)))
            elif resource >= self.v_det[-1, k] + r_t[k]:
                M_t[k] = self.v_det[-1, k] + r_t[k]
            elif self.v_det[-1, k] < resource:  # NOTE: paper also checks < self.v_det[-1,k] + r_t[k] here, but this is unnecessary due to the previous elif
                M_t[k] = choice([self.v_det[-1, k], resource])
            else:
                M_t[k] = resource

            resource -= M_t[k]

        M_t /= sum(M_t)  # normalize in case not all resources have been allocated

        self.r = np.vstack((self.r, r_t))
        self.M = np.vstack((self.M, M_t))
        self.t += 1

        if self.m is not None:
            M_t = make_discrete_allocation(M_t, self.m)

        return M_t

    def update(self, X_t: np.ndarray):
        """
        Update the algorithm parameters based on this round's payoff
        :param X_t: payoff for the current round by battlefield
        """
        self.X = np.vstack((self.X, X_t))

        # NOTE: paper uses i-1 for v_det. Since we haven't inserted v_det_t we don't need to worry about this offset
        s_det_t = np.sum(np.maximum(self.M - self.v_det, 0), axis=0, where=(self.v_det > 0))
        s_prob_t = np.sum(self.M, axis=0, where=(self.M <= self.v_det))

        x_prob_t = np.sum(self.X, axis=0, where=(self.M <= self.v_det))

        eps_t = (self.t ** -3) * (self.K ** -1)
        zeta_t = (sqrt(1 / 2) + sqrt(1 / 2 - log(eps_t))) ** 2

        v_det_t = np.amax(self.M, axis=0, initial=0, where=(self.X == 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            v_prob_t = np.where(s_prob_t > 0, np.power(np.sqrt(zeta_t / (2 * s_prob_t)) + np.sqrt((zeta_t / (2 * s_prob_t)) + (x_prob_t / s_prob_t)), -2), 0)

        self.s_det = np.vstack((self.s_det, s_det_t))
        self.v_det = np.vstack((self.v_det, v_det_t))
        self.v_prob = np.vstack((self.v_prob, v_prob_t))
