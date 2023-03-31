import random
from math import comb
from itertools import combinations_with_replacement

import numpy as np

from cucb_dra import CUCB_DRA
from edge import Edge
from mara import MARA
from optimistic_allocation import Optimistic_Allocation
from pdgraph import coordinate_to_index, allocation_by_id
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import expected_payoff, estimate_best_payoff, best_possible_payoff, supremum_payoff


if __name__ == '__main__':
    battlefields = [5]
    resources = [10, 15, 20]

    estimate_expected_payoff = False
    A_sample_size = None
    B_sample_size = None

    track_dfs = True
    chunksize = 32


    algorithms = {
        MARA: dict(c=2.5),
        CUCB_DRA: dict(),
        Edge: dict(gamma=0.5)
    }

    for K in battlefields:
        for A_resources, B_resources in combinations_with_replacement(resources, 2):
            matchups = []
            print(f'Battlefields: {K}\nA resources: {A_resources}\nB resources: {B_resources}')

            for A_alg, A_kwargs in algorithms.items():
                for B_alg, B_kwargs in algorithms.items():
                    matchups.append((A_alg(K, A_resources, **A_kwargs), B_alg(K, B_resources, **B_kwargs)))
