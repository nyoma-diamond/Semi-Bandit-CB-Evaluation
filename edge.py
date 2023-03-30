from math import comb
import random

import numpy as np
import scipy.sparse as sci

from pdgraph import allocation_by_id


class Edge:
    """
    Edge algorithm from Vu et al. (https://doi.org/10.1109/CDC40024.2019.9029186 and https://doi.org/10.48550/arXiv.1909.04912)
    This is a modified implementation of the code available at https://github.com/dongquan11/BanditColonelBlotto
    """

    def __init__(self, n: int, m: int, gamma: float):
        """
        Edge algorithm
        :param n: battlefields
        :param m: resources available to the algorithm (player)
        :param gamma: probability of exploration
        """
        self.n = n
        self.m = m

        self.E = int(2 * (m + 1) + (n - 2) * (m + 1) * (m + 2) / 2)
        self.N = 2 + (n - 1) * (m + 1)
        self.P = comb(n + m - 1, n - 1)

        self.prev_path = None

        self.Layer_s = np.zeros(self.N)
        self.Vertical_s = np.zeros(self.N)
        self.Children_s = []
        self.Ancestor_s = []

        for u in range(self.N):
            self.Layer_s[u] = self.Layer(u)
            self.Vertical_s[u] = self.Vertical(u)
            self.Children_s.append(list(self.Children(u)))
            self.Ancestor_s.append(list(self.Ancestors(u)))
        self.Layer_s = self.Layer_s.astype(int)
        self.Vertical_s = self.Vertical_s.astype(int)

        self.node_edge = sci.lil_matrix((self.N, self.N))
        self.edge_node = np.zeros((self.E, 2))
        for u1 in range(self.N):
            for u2 in self.Children(u1):
                if u1 < self.N - 1:
                    self.node_edge[u1, u2] = int(self.node_to_edge(u1, u2))
                    self.edge_node[int(self.node_to_edge(u1, u2))] = [int(u1), int(u2)]

        # we always use uniform exploration
        self.w = np.ones(self.E)
        self.H = self.update_H()

        self.C_explore = self.coocurence_mat()

        eigenval = np.sort(
            np.round(np.linalg.eigvalsh(self.C_explore), 8))  # unclear why this is rounded; kept to prevent problems
        for i in range(self.E):
            if np.real(eigenval[i]) + 0 > 0:
                lambda_min = np.real(eigenval[i])
                break

        self.Prob_explore = []
        for u in range(self.N - 1):
            prob = []
            for k in self.Children_s[u]:
                prob = np.append(prob,
                                 [self.w[int(self.node_edge[u, k])] * self.H[k, self.N - 1] / self.H[u, self.N - 1]])
            self.Prob_explore.append(list(prob))

        self.gamma = gamma
        self.eta = self.gamma * lambda_min / n

    def Layer(self, u):
        if u == 0:
            return int(0)
        elif u == self.N - 1:
            return int(self.n)
        else:
            return int(np.floor((u - 1) / (self.m + 1)) + 1)

    def Vertical(self, u):
        if u == 0:
            Vertical = 0
        elif u == self.N - 1:
            Vertical = int(self.m)
        elif np.remainder(u, self.m + 1) == 0:
            Vertical = self.m
        else:
            Vertical = np.remainder(u, self.m + 1) - 1
        return Vertical

    def Children(self, u):
        if u == 0:
            children = np.arange(1, self.m + 2)
        elif u >= self.N - 1 - (self.m + 1):
            children = np.array([self.N - 1])
        else:
            temp = range(self.m + 1 - self.Vertical(u))
            children = (u + self.m + 1) * np.ones(self.m + 1 - self.Vertical(u))
            children = children + temp
        return children.astype(int)

    def Ancestors(self, u):
        if u == self.N - 1:
            ancestor = range(self.N - 1)
        elif self.Layer(u) <= 1:
            ancestor = [0]
        else:
            ancestor = [0]
            for i in range(1, u - self.m):
                if self.Vertical(i) <= self.Vertical(u):
                    ancestor.append(i)
        return ancestor

    def node_to_edge(self, u1, u2):
        if u1 == 0:
            edge = u2 - 1
        elif u2 == self.N - 1:
            edge = self.E - (self.N - 1 - u1)
        else:
            edge = int(self.m + 1
                       + (self.Layer(u1) - 1) * (self.m + 2) * (self.m + 1) / 2
                       + (2 * (self.m + 1) - self.Vertical(u1) + 1) * self.Vertical(u1) / 2
                       + (self.Vertical(u2) - self.Vertical(u1)))
        return edge

    def update_H(self):
        H = np.identity(self.N)
        for j in np.flip(range(self.N), axis=0):
            for i in np.flip(self.Ancestor_s[j], axis=0):
                for k in self.Children_s[i]:
                    H[i, j] = H[i, j] + self.w[int(self.node_edge[i, k])] * H[k, j]

        return H

    def single_prob(self, e):
        single_prob = self.H[0, int(self.edge_node[e, 0])] * self.w[e] * self.H[int(self.edge_node[e, 1]), self.N - 1] / self.H[0, self.N - 1]
        return single_prob

    def coocurence_mat(self):
        mat = np.zeros((self.E, self.E))
        for e_1 in range(self.E):
            mat[e_1, e_1] = self.single_prob(e_1)
            for e_2 in range(e_1 + 1, self.E):
                mat[e_1, e_2] = (self.H[0, int(self.edge_node[e_1, 0])]
                                 * self.w[e_1]
                                 * self.H[int(self.edge_node[e_1, 1]), int(self.edge_node[e_2, 0])]
                                 * self.w[e_2]
                                 * self.H[int(self.edge_node[e_2, 1]), self.N - 1]
                                 / self.H[0, self.N - 1])
                mat[e_2, e_1] = mat[e_1, e_2]
        return mat

    def exploit(self):
        node_k_1 = 0
        chosen_path = np.array([0])
        while len(chosen_path) <= self.n:
            prob = np.array([])
            for k in self.Children_s[node_k_1]:
                prob = np.append(prob, [
                    self.w[int(self.node_edge[node_k_1, k])] * self.H[k, self.N - 1] / self.H[node_k_1, self.N - 1]])
            node_k = np.random.choice(self.Children_s[node_k_1], p=prob)
            chosen_path = np.append(chosen_path, node_k)
            node_k_1 = node_k

        return chosen_path

    def explore(self):
        node_k_1 = 0
        chosen_path = np.array([0])
        while len(chosen_path) <= self.n:
            node_k = np.random.choice(self.Children_s[node_k_1], p=self.Prob_explore[node_k_1])
            chosen_path = np.append(chosen_path, node_k)
            node_k_1 = node_k
        return chosen_path

    def allo(self, e):
        alloc = np.zeros(2)
        alloc[1] = self.Layer(self.edge_node[e, 1])
        if self.edge_node[e, 0] == 0 or self.edge_node[e, 1] == self.N - 1:
            alloc[0] = self.edge_node[e, 1] - self.edge_node[e, 0] - 1
        else:
            alloc[0] = self.edge_node[e, 1] - self.edge_node[e, 0] - (self.m + 1)
        return alloc

    def bin_path(self, p):
        path_temp = np.zeros(shape=(self.n, 2))
        for i in range(self.n):
            path_temp[i] = [p[i], p[i + 1]]

        bin_paths = np.zeros(self.E)
        for j in range(self.E):
            if any(np.equal(path_temp, self.edge_node[j]).all(1)) == 1:
                bin_paths[j] = 1
        return bin_paths

    def generate_decision(self):
        self.H = self.update_H()
        if random.random() > self.gamma:  # Exploit (beta == 0 check)
            print('exploiting')
            path = self.exploit()
        else:  # Explore
            print('exploring')
            path = self.explore()

        self.prev_path = path
        allocation = np.apply_along_axis(player.allo, 1, np.argwhere(player.bin_path(path) == 1))

        return allocation[allocation[:, 1].argsort()][:, 0]  # sorting just in case the ordering gets messed up

    def update(self, loss):
        # C = (1-self.gamma)*self.coocurence_mat()  + self.gamma * self.C_explore
        C = self.coocurence_mat()
        est_loss = np.asarray(loss * (np.matmul(np.linalg.pinv(C), self.bin_path(self.prev_path)))).flatten()
        self.w = self.w * np.exp(-self.eta * est_loss)


if __name__ == '__main__':
    battlefields = 5  # battlefields
    resources = 15  # resources
    opp_resources = 15

    opp_num_decisions = comb(battlefields + opp_resources - 1, battlefields - 1)

    player = Edge(battlefields, resources, 0.5)

    for _ in range(30):
        print()
        allocation = player.generate_decision()
        print(f'Player\'s allocation: {allocation} (total: {sum(allocation)})')

        opp_allocation = np.asarray(
            allocation_by_id(random.randint(0, opp_num_decisions - 1), battlefields, opp_resources))
        print('Opponent\'s allocation:', opp_allocation)

        result = np.greater(allocation, opp_allocation)
        print('Result:', result)
        print('Payoff:', sum(result))
        print('Loss:', (battlefields - sum(result)) / battlefields)

        player.update((battlefields - sum(result)) / battlefields)
