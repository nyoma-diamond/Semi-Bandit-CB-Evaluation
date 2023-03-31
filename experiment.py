import random
from math import comb
from itertools import combinations_with_replacement

import numpy as np

from cucb_dra import CUCB_DRA
from edge import Edge
from mara import MARA
from optimistic_allocation import Optimistic_Allocation
from pdgraph import coordinate_to_index, allocation_by_id, compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import expected_payoff, estimate_best_payoff, best_possible_payoff, supremum_payoff


if __name__ == '__main__':
    battlefields = [5]
    resources = [10, 15, 20]
    T = 10

    estimate_expected_payoff = False
    A_sample_size = None
    B_sample_size = None

    track_dfs = False
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

            A_graph = build_adjacency_matrix(K, A_resources)
            B_graph = build_adjacency_matrix(K, B_resources)

            # print('A nodes:', A_graph.shape[0], '| A edges:', (A_graph >= 0).sum())
            # print('B nodes:', B_graph.shape[0], '| B edges:', (B_graph >= 0).sum())

            A_num_decisions = comb(K+A_resources-1, K-1)
            B_num_decisions = comb(K+B_resources-1, K-1)

            d_A = coordinate_to_index((K, A_resources), K, A_resources)
            d_B = coordinate_to_index((K, B_resources), K, B_resources)

            for A_alg, A_kwargs in algorithms.items():
                for B_alg, B_kwargs in algorithms.items():
                    print('Player A:', A_alg.__name__)
                    print('Player B:', B_alg.__name__)

                    player_A = A_alg(K, A_resources, **A_kwargs)
                    player_B = B_alg(K, B_resources, **B_kwargs)

                    for _ in range(T):
                        A_decision = player_A.generate_decision()
                        B_decision = player_B.generate_decision()

                        # print('A decision:', A_decision)
                        # print('B decision:', B_decision)

                        A_result = np.greater(A_decision, B_decision)
                        B_result = ~A_result

                        # print('A payoff:', A_result.sum())
                        # print('B payoff:', B_result.sum())

                        player_A.update(A_result)
                        player_B.update(B_result)

                        B_bounds = compute_bounds(A_decision, A_result, B_resources, False)
                        A_bounds = compute_bounds(B_decision, B_result, A_resources, True)

                        # print('Bounds on B:', np.array2string(B_bounds.T, prefix='Bounds on B:'))
                        # print('Bounds on A:', np.array2string(A_bounds.T, prefix='Bounds on A:'))

                        pruned_B = prune_dead_ends(build_adjacency_matrix(K, B_resources, B_bounds), prune_unreachable=track_dfs)
                        pruned_A = prune_dead_ends(build_adjacency_matrix(K, A_resources, A_bounds), prune_unreachable=track_dfs)

                        B_possible_decisions = find_paths_allocations(pruned_B, d_B, K, B_resources, track_progress=track_dfs)
                        A_possible_decisions = find_paths_allocations(pruned_A, d_A, K, A_resources, track_progress=track_dfs)

                        if estimate_expected_payoff:
                            if A_sample_size is None:
                                A_decisions = find_paths_allocations(A_graph, d_A, K, A_resources, track_progress=track_dfs)
                            else:
                                A_decisions = [allocation_by_id(id, K, A_resources) for id in random.choices(range(A_num_decisions), k=A_sample_size)]

                            A_expected_payoff = expected_payoff(A_decisions, np.expand_dims(B_decision, axis=0), win_draws=False, chunksize=chunksize)

                            A_expected_regret = A_expected_payoff - A_result.sum()
                            A_estimated_expected_payoff = expected_payoff(A_decisions, B_possible_decisions, win_draws=False, chunksize=chunksize)

                            A_estimated_expected_regret = A_estimated_expected_payoff - A_result.sum()

                        A_best_payoff = best_possible_payoff(B_decision, A_resources, win_draws=False)
                        A_best_regret = A_best_payoff - A_result.sum()

                        A_estimated_best_payoff = estimate_best_payoff(B_possible_decisions, A_resources, win_draws=False, chunksize=chunksize)
                        A_estimated_best_regret = A_estimated_best_payoff - A_result.sum()

                        A_supremum_payoff = supremum_payoff(B_possible_decisions, A_resources, win_draws=False, chunksize=chunksize)
                        A_supremum_regret = A_supremum_payoff - A_result.sum()



                        if estimate_expected_payoff:
                            if B_sample_size is None:
                                B_decisions = find_paths_allocations(B_graph, d_B, K, B_resources, track_progress=track_dfs)
                            else:
                                B_decisions = [allocation_by_id(id, K, B_resources) for id in random.choices(range(B_num_decisions), k=B_sample_size)]

                            B_expected_payoff = expected_payoff(B_decisions, np.expand_dims(A_decision, axis=0), win_draws=True, chunksize=chunksize)

                            B_expected_regret = B_expected_payoff - B_result.sum()
                            B_estimated_expected_payoff = expected_payoff(B_decisions, A_possible_decisions, win_draws=True, chunksize=chunksize)

                            B_estimated_expected_regret = B_estimated_expected_payoff - B_result.sum()

                        B_best_payoff = best_possible_payoff(A_decision, B_resources, win_draws=True)
                        B_best_regret = B_best_payoff - B_result.sum()

                        B_estimated_best_payoff = estimate_best_payoff(A_possible_decisions, B_resources, win_draws=True, chunksize=chunksize)
                        B_estimated_best_regret = B_estimated_best_payoff - B_result.sum()

                        B_supremum_payoff = supremum_payoff(A_possible_decisions, B_resources, win_draws=True, chunksize=chunksize)
                        B_supremum_regret = B_supremum_payoff - B_result.sum()