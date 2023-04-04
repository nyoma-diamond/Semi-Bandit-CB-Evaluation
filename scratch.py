import random
from math import comb

import numpy as np

np.random.seed(42)
random.seed(42)

from pdgraph import allocation_by_id, compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import expected_payoff, estimate_best_payoff, best_possible_payoff, supremum_payoff


if __name__ == '__main__':
    battlefields = 15
    N_A = 30
    N_B = 20
    N_A_est = N_A
    N_B_est = N_B

    estimate_expected_payoff = False
    A_sample_size = None
    B_sample_size = None

    track_dfs = True
    chunksize = 32


    print('===== Parameters =====')

    print('Battlefields:', battlefields)
    print('A\'s resources:', N_A, '| B\'s estimate of A\'s resources:', N_A_est)
    print('B\'s resources:', N_B, '| A\'s estimate of B\'s resources:', N_B_est)

    print(f'A reference decision set: {"True" if A_sample_size is None else "Sampled"} decision set{"" if A_sample_size is None else f" of size {A_sample_size}"}')
    print(f'B reference decision set: {"True" if B_sample_size is None else "Sampled"} decision set{"" if B_sample_size is None else f" of size {B_sample_size}"}')


    print('\n===== Graph information =====')

    print('Building base adjacency matrices...')

    adj_mat_A = build_adjacency_matrix(battlefields, N_A)
    adj_mat_B = build_adjacency_matrix(battlefields, N_B)

    print('A nodes:',adj_mat_A.shape[0], '| A edges:',(adj_mat_A >= 0).sum())
    print('B nodes:',adj_mat_B.shape[0], '| B edges:',(adj_mat_B >= 0).sum())

    A_num_decisions = comb(battlefields+N_A-1, battlefields-1)
    B_num_decisions = comb(battlefields+N_B-1, battlefields-1)

    print('A decisions:', A_num_decisions)
    print('B decisions:', B_num_decisions)



    print('\n===== Example round =====')

    A_play = allocation_by_id(random.randint(0, A_num_decisions-1), battlefields, N_A)
    B_play = allocation_by_id(random.randint(0, B_num_decisions-1), battlefields, N_B)

    print('A allocation:', A_play)
    print('B allocation:', B_play)

    A_result = np.greater(A_play, B_play)
    B_result = np.invert(A_result)

    print('A result:', A_result)
    print('B result:', B_result)

    print('A payoff:', A_result.sum())
    print('B payoff:', B_result.sum())



    print('\n===== Possible opponent play =====')

    print('Computing bounds...')

    B_bounds = compute_bounds(A_play, A_result, N_B, False)
    A_bounds = compute_bounds(B_play, B_result, N_A, True)

    print('Bounds on B:', np.array2string(B_bounds.T, prefix='Bounds on B:'))
    print('Bounds on A:', np.array2string(A_bounds.T, prefix='Bounds on A:'))

    print('Building possible decision (pruned) adjacency matrices...')

    pruned_A = prune_dead_ends(build_adjacency_matrix(battlefields, N_A_est, A_bounds), prune_unreachable=track_dfs)
    pruned_B = prune_dead_ends(build_adjacency_matrix(battlefields, N_B_est, B_bounds), prune_unreachable=track_dfs)

    print('\nFinding all possible decisions for player A...', flush=True)
    A_possible_decisions = find_paths_allocations(pruned_A, battlefields, N_A_est, track_progress=track_dfs)
    print('Possible decisions B thinks A could have played:', len(A_possible_decisions))

    print('\nFinding all possible decisions for player B...', flush=True)
    B_possible_decisions = find_paths_allocations(pruned_B, battlefields, N_B_est, track_progress=track_dfs)
    print('Possible decisions A thinks B could have played:', len(B_possible_decisions))



    print('\n===== Expected payoff/regret =====')

    print('--- Player A ---', flush=True)

    if estimate_expected_payoff:
        if A_sample_size is None:
            A_decisions = find_paths_allocations(adj_mat_A, battlefields, N_A, track_progress=track_dfs)
        else:
            A_decisions = [allocation_by_id(id, battlefields, N_A) for id in random.choices(range(A_num_decisions), k=A_sample_size)]

        A_expected_payoff = expected_payoff(A_decisions, np.expand_dims(B_play, axis=0), win_draws=False, chunksize=chunksize)

        A_expected_regret = A_expected_payoff - A_result.sum()
        A_estimated_expected_payoff = expected_payoff(A_decisions, B_possible_decisions, win_draws=False, chunksize=chunksize)

        A_estimated_expected_regret = A_estimated_expected_payoff - A_result.sum()

    A_best_payoff = best_possible_payoff(B_play, N_A, win_draws=False)
    A_best_regret = A_best_payoff - A_result.sum()

    A_estimated_best_payoff = estimate_best_payoff(B_possible_decisions, N_A, win_draws=False, chunksize=chunksize)
    A_estimated_best_regret = A_estimated_best_payoff - A_result.sum()

    A_supremum_payoff = supremum_payoff(B_possible_decisions, N_A, win_draws=False, chunksize=chunksize)
    A_supremum_regret = A_supremum_payoff - A_result.sum()

    print('Round payoff:', A_result.sum())

    print('\nBest possible payoff:', A_best_payoff)
    if estimate_expected_payoff:
        print('Expected payoff:', A_expected_payoff)

    print('\nEstimated best payoff:', A_estimated_best_payoff)
    print('Supremum payoff:', A_supremum_payoff)
    if estimate_expected_payoff:
        print('Estimated expected payoff:', A_estimated_expected_payoff)

    print('\nEstimated best possible regret:', A_estimated_best_regret)
    print('Supremum regret:', A_supremum_regret)
    if estimate_expected_payoff:
        print('Estimated expected regret:', A_estimated_expected_regret)


    print('\n--- Player B ---', flush=True)

    if estimate_expected_payoff:
        if B_sample_size is None:
            B_decisions = find_paths_allocations(adj_mat_B, battlefields, N_B, track_progress=track_dfs)
        else:
            B_decisions = [allocation_by_id(id, battlefields, N_B) for id in random.choices(range(B_num_decisions), k=B_sample_size)]

        B_expected_payoff = expected_payoff(B_decisions, np.expand_dims(A_play, axis=0), win_draws=True, chunksize=chunksize)

        B_expected_regret = B_expected_payoff - B_result.sum()
        B_estimated_expected_payoff = expected_payoff(B_decisions, A_possible_decisions, win_draws=True, chunksize=chunksize)

        B_estimated_expected_regret = B_estimated_expected_payoff - B_result.sum()

    B_best_payoff = best_possible_payoff(A_play, N_B, win_draws=True)
    B_best_regret = B_best_payoff - B_result.sum()

    B_estimated_best_payoff = estimate_best_payoff(A_possible_decisions, N_B, win_draws=True, chunksize=chunksize)
    B_estimated_best_regret = B_estimated_best_payoff - B_result.sum()

    B_supremum_payoff = supremum_payoff(A_possible_decisions, N_B, win_draws=True, chunksize=chunksize)
    B_supremum_regret = B_supremum_payoff - B_result.sum()

    print('Round payoff:', B_result.sum())

    print('\nBest possible payoff:', B_best_payoff)
    if estimate_expected_payoff:
        print('Expected payoff:', B_expected_payoff)

    print('\nEstimated best payoff:', B_estimated_best_payoff)
    print('Supremum payoff:', B_supremum_payoff)
    if estimate_expected_payoff:
        print('Estimated expected payoff:', B_estimated_expected_payoff)

    print('\nEstimated best possible regret:', B_estimated_best_regret)
    print('Supremum regret:', B_supremum_regret)
    if estimate_expected_payoff:
        print('Estimated expected regret:', B_estimated_expected_regret)