import random
import multiprocessing as mp
from functools import partial
from time import sleep

import numpy as np
from tqdm import tqdm

from pdgraph import build_adjacency_matrix, find_paths_allocations, compute_expected_payoff


if __name__ == '__main__':
    battlefields = 6
    N_A = 20
    N_B = 15

    d_A = (battlefields, N_A)
    d_B = (battlefields, N_B)

    print('Building base adjacency matrices...')

    adj_mat_A = build_adjacency_matrix(battlefields, N_A)
    adj_mat_B = build_adjacency_matrix(battlefields, N_B)

    print('Finding all possible decisions...')

    A_decisions = find_paths_allocations(adj_mat_A, d_A)
    B_decisions = find_paths_allocations(adj_mat_B, d_B)

    print('A decisions:', len(A_decisions))
    print('B decisions:', len(B_decisions))

    A_play = random.choice(A_decisions)
    B_play = random.choice(B_decisions)

    print('A allocation:', A_play)
    print('B allocation:', B_play)

    A_result = np.greater(A_play, B_play)
    B_result = np.invert(A_result)

    print('A payoff:', A_result.sum())
    print('B payoff:', B_result.sum())

    print('Computing bounds...')

    A_bounds, B_bounds = [], []

    for allocation, win in zip(A_play, A_result):
        if win:
            B_bounds.append((0, allocation - 1))
        else:
            B_bounds.append((allocation, N_B - (np.argwhere(B_result).sum() - allocation))) # note that A losses are B wins

    for allocation, win in zip(B_play, B_result):
        if win:
            A_bounds.append((0, allocation))
        else:
            A_bounds.append((allocation+1, N_A + 1 - A_result.sum() - (np.argwhere(A_result).sum()-allocation))) # note that B losses are A wins

    print('Building possible decision (pruned) adjacency matrices...')

    pruned_A = build_adjacency_matrix(battlefields, N_A, A_bounds)
    pruned_B = build_adjacency_matrix(battlefields, N_B, B_bounds)

    print('Finding all possible decisions for round...')

    A_possible_decisions = find_paths_allocations(pruned_A, d_A)
    B_possible_decisions = find_paths_allocations(pruned_B, d_B)


    A_expected_payoff = 0
    B_expected_payoff = 0


    print('Computing expected payoff for player A...')
    with tqdm(total=len(A_decisions)*len(B_possible_decisions), unit_scale=True) as pbar:
        with mp.Pool() as pool:
            for i, partial_payoff in enumerate(pool.imap_unordered(partial(compute_expected_payoff,
                                                                           opp_decisions=A_decisions,
                                                                           win_draws=False,
                                                                           divide=False),
                                                                   B_possible_decisions,
                                                                   chunksize=32)):
                A_expected_payoff += partial_payoff
                pbar.update(len(A_decisions))
                pbar.set_postfix_str(f'Expected payoff: {A_expected_payoff / (len(A_decisions)*(i+1))}')

    A_expected_payoff /= (len(A_decisions)*len(B_possible_decisions))
    sleep(0.1) # fudge to make sure printout doesn't get messed up
    print('Expected payoff for player A:', A_expected_payoff)


    print('Computing expected payoff for player B...')
    with tqdm(total=len(B_decisions)*len(A_possible_decisions), unit_scale=True) as pbar:
        with mp.Pool() as pool:
            for i, partial_payoff in enumerate(pool.imap_unordered(partial(compute_expected_payoff,
                                                                           opp_decisions=B_decisions,
                                                                           win_draws=False,
                                                                           divide=False),
                                                                   A_possible_decisions,
                                                                   chunksize=32)):
                B_expected_payoff += partial_payoff
                pbar.update(len(B_decisions))
                pbar.set_postfix_str(f'Expected payoff: {B_expected_payoff / (len(B_decisions)*(i+1))}')

    B_expected_payoff /= (len(B_decisions)*len(A_possible_decisions))
    sleep(0.1) # fudge to make sure printout doesn't get messed up
    print('Expected payoff for player B:', B_expected_payoff)
