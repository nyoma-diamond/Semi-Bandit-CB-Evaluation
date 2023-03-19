import math
import numpy as np


class optimistic_allocation:
    """
    Optimistic Allocation algorithm from Lattimore et al. (https://arxiv.org/pdf/1406.3840.pdf)
    """

    def __init__(self, n: int, K: int, v_lb: np.ndarray):
        """
        Optimistic Allocation initializer
        :param n: time horizon
        :param K: the number of jobs (battlefields)
        :param v_lb: initial estimated lower bounds on v for each battlefield
        """
        self.K = K
        self.v_lb = np.expand_dims(v_lb, axis=0)

        self.delta = (n*K)**(-2)
        self.v_ub = np.full_like(self.v_lb, np.inf)

        self.M = np.empty(shape=(0,K))
        self.X = np.empty(shape=(0,K))
        self.w = np.empty(shape=(0,K))

        self.t = 0


    def play_decision(self, M_t):
        """
        Play a decision
        :param M_t: decision to make
        """
        self.M = np.append(self.M, np.expand_dims(M_t, axis=0), axis=0)
        self.t += 1


    def generate_decision(self):
        """
        Generate an allocation decision
        :return: the allocation decision
        """
        M_t = np.zeros(shape=self.K)

        cur_v_lb = self.v_lb[-1]

        for i in range(self.K):
            args = np.argwhere(M_t == 0)
            k = args[cur_v_lb[args].argmin()]
            M_t[k] = min(cur_v_lb[k], 1-np.sum(M_t))

        return M_t


    def update(self, X_t: np.ndarray):
        """
        Update the algorithm parameters based on this round's payoff
        :param X_t: payoff for the current round by battlefield
        """
        self.X = np.append(self.X, np.expand_dims(X_t, axis=0), axis=0)

        w_t = 1 / (1 - (self.M[-1] / self.v_ub[-1]))
        self.w = np.append(self.w, np.expand_dims(w_t, axis=0), axis=0)

        v_hat_inv_t = np.sum(self.w * self.X, axis=0) / np.sum(self.w * self.M, axis=0)

        R_t = np.amax(self.w, axis=0)

        V_hat_sq_t = np.sum(self.w * self.M / self.v_lb[-1], axis=0)

        epsilon_t = self.f(R_t, V_hat_sq_t)

        v_lb_inv_t = np.minimum(np.reciprocal(self.v_lb[-1]), v_hat_inv_t + epsilon_t)
        v_ub_inv_t = np.maximum(np.reciprocal(self.v_ub[-1]), v_hat_inv_t - epsilon_t)

        self.v_lb = np.append(self.v_lb, np.expand_dims(np.reciprocal(v_lb_inv_t), axis=0), axis=0)
        self.v_ub = np.append(self.v_ub, np.expand_dims(np.reciprocal(v_ub_inv_t), axis=0), axis=0)


    def f(self, R_t, V_sq_t):
        delta_0 = self.delta / (3 * np.square(R_t+1) * np.square(V_sq_t+1))

        return ((R_t+1) / 3) * np.log(2/delta_0) + \
            np.sqrt(2 * np.square(V_sq_t+1) * np.log(2/delta_0) + np.square((R_t+1)/3) * np.square(np.log(2/delta_0)))




if __name__ == '__main__':
    battlefields = 5
    N_A = 30
    N_B = 20

    horizon = 10

    track_dfs = True

    v = np.array([0.1, 0.5, 0.2, 0.3, 0.4], dtype=np.float_)

    player_B = optimistic_allocation(horizon, battlefields, np.array([0.09, 0.4, 0.15, 0.25, 0.3]))

    for i in range(horizon):
        print()
        allocation = player_B.generate_decision()
        print('Player B\'s allocation:', allocation)
        player_B.play_decision(allocation)

        p = np.minimum(1, allocation / v)
        print('Success probabilities:', p)

        X = np.random.binomial(1, p)
        print('Payoff:', X)

        player_B.update(X)
