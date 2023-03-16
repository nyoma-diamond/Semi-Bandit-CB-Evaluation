import random
from math import comb

import numpy as np

from pdgraph import coordinate_to_index, allocation_by_id
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import compute_expected_payoff, compute_expected_best_payoff, best_possible_payoff


if __name__ == '__main__':
    battlefields = 6
    N_A = 20
    N_B = 20
    N_A_est = N_A
    N_B_est = N_B

    use_expected_payoff = False
    A_sample_size = None
    B_sample_size = None

    track_dfs = True


    print('===== Parameters =====')

    print('Battlefields:', battlefields)
    print('A\'s resources:', N_A, '| B\'s estimate of A\'s resources:', N_A_est)
    print('B\'s resources:', N_B, '| A\'s estimate of B\'s resources:', N_B_est)

    print(f'A reference decision set: {"True" if A_sample_size is None else "Sampled"} decision set{"" if A_sample_size is None else f" of size {A_sample_size}"}')
    print(f'B reference decision set: {"True" if B_sample_size is None else "Sampled"} decision set{"" if B_sample_size is None else f" of size {B_sample_size}"}')

    d_A = coordinate_to_index((battlefields, N_A), battlefields, N_A)
    d_B = coordinate_to_index((battlefields, N_B), battlefields, N_B)

    d_A_est = coordinate_to_index((battlefields, N_A_est), battlefields, N_A_est)
    d_B_est = coordinate_to_index((battlefields, N_B_est), battlefields, N_B_est)




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

    A_play = np.asarray(allocation_by_id(random.randint(0, A_num_decisions-1), battlefields, N_A))
    B_play = np.asarray(allocation_by_id(random.randint(0, B_num_decisions-1), battlefields, N_B))

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

    A_bounds, B_bounds = [], []

    for allocation, win in zip(A_play, A_result):
        if win:
            B_bounds.append((0, allocation - 1))
        else:
            B_bounds.append((allocation, N_B - (A_play.sum(where=B_result) - allocation))) # note that A losses are B wins

    for allocation, win in zip(B_play, B_result):
        if win:
            A_bounds.append((0, allocation))
        else:
            A_bounds.append((allocation+1, N_A + 1 - A_result.sum() - (B_play.sum(where=A_result)-allocation))) # note that B losses are A wins

    print('Bounds on B:', B_bounds)
    print('Bounds on A:', A_bounds)

    print('Building possible decision (pruned) adjacency matrices...')

    pruned_A = prune_dead_ends(build_adjacency_matrix(battlefields, N_A_est, A_bounds), prune_unreachable=track_dfs)
    pruned_B = prune_dead_ends(build_adjacency_matrix(battlefields, N_B_est, B_bounds), prune_unreachable=track_dfs)

    print('\nFinding all possible decisions for player A...')
    A_possible_decisions = find_paths_allocations(pruned_A, d_A_est, battlefields, N_A_est, track_progress=track_dfs)
    print('Possible decisions B thinks A could have played:', len(A_possible_decisions))

    print('\nFinding all possible decisions for player B...')
    B_possible_decisions = find_paths_allocations(pruned_B, d_B_est, battlefields, N_B_est, track_progress=track_dfs)
    print('Possible decisions A thinks B could have played:', len(B_possible_decisions))



    print('\n===== Expected payoff/regret =====')

    print('--- Player A ---')

    if use_expected_payoff:
        if A_sample_size is None:
            A_decisions = find_paths_allocations(adj_mat_A, d_A, battlefields, N_A, track_progress=track_dfs)
        else:
            A_decisions = [allocation_by_id(id, battlefields, N_A) for id in random.choices(range(A_num_decisions), k=A_sample_size)]

        A_expected_payoff = compute_expected_payoff(A_decisions, B_possible_decisions, win_draws=False, chunksize=16)
        A_expected_regret = A_expected_payoff - A_result.sum()

    A_expected_best_payoff = compute_expected_best_payoff(B_possible_decisions, N_A, win_draws=False)
    A_best_possible_payoff = best_possible_payoff(list(B_play), N_A, win_draws=False)

    A_observable_regret = A_expected_best_payoff - A_result.sum()
    A_absolute_regret = A_best_possible_payoff - A_result.sum()

    print('Actual payoff:', A_result.sum())
    if use_expected_payoff:
        print('Expected payoff:', A_expected_payoff)
    print('Expected best payoff:', A_expected_best_payoff)
    print('Absolute best possible payoff:', A_best_possible_payoff, '\n')

    if use_expected_payoff:
        print('Expected regret:', A_expected_regret)
    print('Observable regret:', A_observable_regret)
    print('Absolute regret:', A_absolute_regret)


    print('\n--- Player B ---')

    if use_expected_payoff:
        if B_sample_size is None:
            B_decisions = find_paths_allocations(adj_mat_B, d_B, battlefields, N_B, track_progress=track_dfs)
        else:
            B_decisions = [allocation_by_id(id, battlefields, N_B) for id in random.choices(range(B_num_decisions), k=B_sample_size)]

        B_expected_payoff = compute_expected_payoff(B_decisions, A_possible_decisions, win_draws=True, chunksize=16)
        B_expected_regret = B_expected_payoff - B_result.sum()

    B_expected_best_payoff = compute_expected_best_payoff(A_possible_decisions, N_B, win_draws=True)
    B_best_possible_payoff = best_possible_payoff(list(A_play), N_B, win_draws=True)

    B_observable_regret = B_expected_best_payoff - B_result.sum()
    B_absolute_regret = B_best_possible_payoff - B_result.sum()

    print('Actual payoff:', B_result.sum())
    if use_expected_payoff:
        print('Expected payoff:', B_expected_payoff)
    print('Expected best payoff:', B_expected_best_payoff)
    print('Absolute best possible payoff:', B_best_possible_payoff, '\n')

    if use_expected_payoff:
        print('Expected regret:', B_expected_regret)
    print('Observable regret:', B_observable_regret)
    print('Absolute regret:', B_absolute_regret)