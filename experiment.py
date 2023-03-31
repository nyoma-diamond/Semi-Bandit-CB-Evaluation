import random
from math import comb

import numpy as np

random.seed(42)
np.random.seed(42)

from pdgraph import coordinate_to_index, allocation_by_id, compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import expected_payoff, estimate_best_payoff, best_possible_payoff, supremum_payoff

if __name__ == '__main__':
    estimate_expected_payoff = False
    A_sample_size = None
    B_sample_size = None

    track_dfs = True
    chunksize = 32

    games = np.load('games.npy', allow_pickle=True)[()]

    for K in games.keys():
        for A_resources, B_resources in games[K].keys():
            A_graph = build_adjacency_matrix(K, A_resources)
            B_graph = build_adjacency_matrix(K, B_resources)

            A_num_decisions = comb(K+A_resources-1, K-1)
            B_num_decisions = comb(K+B_resources-1, K-1)

            d_A = coordinate_to_index((K, A_resources), K, A_resources)
            d_B = coordinate_to_index((K, B_resources), K, B_resources)

            for (A_alg, B_alg), game in games[K][(A_resources, B_resources)].items():
                for A_decision, B_decision, A_result, B_result in zip(game['A']['Decisions'], game['B']['Decisions'], game['A']['Results'], game['B']['Results']):
                    B_bounds = compute_bounds(A_decision, A_result, B_resources, False)
                    A_bounds = compute_bounds(B_decision, B_result, A_resources, True)

                    # print('Bounds on B:', np.array2string(B_bounds.T, prefix='Bounds on B:'))
                    # print('Bounds on A:', np.array2string(A_bounds.T, prefix='Bounds on A:'))

                    pruned_B = prune_dead_ends(build_adjacency_matrix(K, B_resources, B_bounds), prune_unreachable=track_dfs)
                    pruned_A = prune_dead_ends(build_adjacency_matrix(K, A_resources, A_bounds), prune_unreachable=track_dfs)

                    print('Finding believed possible decisions for B and A...')

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

                    print('Computing best possible payoff for A...')
                    A_best_payoff = best_possible_payoff(B_decision, A_resources, win_draws=False)
                    A_best_regret = A_best_payoff - A_result.sum()

                    print('Computing best payoff estimate for A...')
                    A_estimated_best_payoff = estimate_best_payoff(B_possible_decisions, A_resources, win_draws=False, chunksize=chunksize)
                    A_estimated_best_regret = A_estimated_best_payoff - A_result.sum()

                    print('Computing supremum payoff for A...')
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

                    print('Computing best possible payoff for B...')
                    B_best_payoff = best_possible_payoff(A_decision, B_resources, win_draws=True)
                    B_best_regret = B_best_payoff - B_result.sum()

                    print('Computing best payoff estimate for B...')
                    B_estimated_best_payoff = estimate_best_payoff(A_possible_decisions, B_resources, win_draws=True, chunksize=chunksize)
                    B_estimated_best_regret = B_estimated_best_payoff - B_result.sum()

                    print('Computing supremum payoff for B...')
                    B_supremum_payoff = supremum_payoff(A_possible_decisions, B_resources, win_draws=True, chunksize=chunksize)
                    B_supremum_regret = B_supremum_payoff - B_result.sum()