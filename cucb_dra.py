import math
import random

import numpy as np

from pdgraph import build_adjacency_matrix, coordinate_to_index, find_paths_allocations, allocation_by_id


class Oracle():
    def __init__(self, K: int, Q: int):
        # self.alpha = alpha
        # self.beta = beta

        adj_mat = build_adjacency_matrix(K, Q)
        d = coordinate_to_index((K, Q), K, Q)

        self.action_set = find_paths_allocations(adj_mat, d, K, Q, track_progress=True)

        def expand(arr):
            new = np.zeros((K, Q+1))
            new[np.arange(0, arr.size), arr] = 1
            return new

        self.expanded_action_set = np.apply_along_axis(expand, 1, self.action_set)




    def r(self, a: np.ndarray, D: np.ndarray):
        """
        The (expected) reward obtained for a given allocation based on a provided success distribution
        I.e., we assume that the likelihood of success for battlefield k = a_k * D_k
        :param a: allocations to test
        :param D: probabilities of success per unit allocated
        :return: expected reward
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.nan_to_num(np.minimum(a * D, 1), nan=0, posinf=0)

        if p.ndim == 1:
            p = np.expand_dims(p, axis=0)

        return np.sum(np.sum(p, axis=1), axis=1)

    def opt(self, D: np.ndarray):
        """
        Get the expected reward for the optimal allocation based on a provided success distribution
        :param D: probabilities of success per unit allocated
        :return: the index of the decision providing the optimal expected reward, value of the optimal expected reward
        """
        # TODO: supremum from possible exploration?
        expected_rewards = self.r(self.expanded_action_set, D)
        max_index = np.argmax(expected_rewards)
        return max_index, expected_rewards[max_index]

    def generate_decision(self, D):
        return self.action_set[self.opt(D)[0]]


class CUCB_DRA():
    """
    Online CUCB_DRA algorithm from Zuo and Joe-Wong (https://doi.org/10.1109/CISS50987.2021.9400228)
    """

    def __init__(self, Q: int, K: int):
        self.Q = Q
        self.K = K

        self.oracle = Oracle(K, Q)

        self.T = np.zeros(shape=(K, Q + 1), dtype=np.float_)  # number of times each arm has been played
        self.mu_hat = np.zeros(shape=(K, Q + 1), dtype=np.float_)  # empirical mean of reward function

        self.plays = np.empty(shape=(0, K), dtype=np.int_)

        self.t = 1

    def generate_decision(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.nan_to_num(np.sqrt(3 * math.log(self.t) / (2 * self.T)), nan=np.inf, posinf=np.inf)  # Confidence radius

        mu_bar = self.mu_hat + rho  # upper confidence bound

        allocation = self.oracle.generate_decision(mu_bar)

        return allocation

    def play_decision(self, decision):
        self.t += 1
        self.T[np.arange(decision.size), decision] += 1
        self.plays = np.append(self.plays, np.expand_dims(decision, axis=0), axis=0)

    def update(self, reward):
        decision = self.plays[-1]

        T_ka = self.T[np.arange(decision.size), decision]
        mu_hat_ka = self.mu_hat[np.arange(decision.size), decision]

        self.mu_hat[np.arange(decision.size), decision] = mu_hat_ka - (reward - mu_hat_ka) / T_ka


if __name__ == '__main__':
    battlefields = 5
    Q = 15
    Q_opp = 20

    opp_num_decisions = math.comb(battlefields + Q_opp - 1, battlefields - 1)

    player = CUCB_DRA(Q, battlefields)

    for _ in range(20):
        print()
        allocation = player.generate_decision()
        print('Player\'s allocation:', allocation)
        player.play_decision(allocation)

        opp_allocation = np.asarray(allocation_by_id(random.randint(0, opp_num_decisions - 1), battlefields, Q_opp))
        print('Opponent\'s allocation:', opp_allocation)

        result = np.greater(allocation, opp_allocation)
        print('Result:', result)
        print('Payoff:', sum(result))

        player.update(result)
