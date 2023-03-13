import random
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm

from pdgraph import build_adjacency_matrix, find_paths_allocations, build_payoff_matrix


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

    print('Computing payoff graphs for player A...')
    with mp.Pool() as pool:
        A_payoff_mats = list(tqdm(pool.imap_unordered(partial(build_payoff_matrix, adj_mat=adj_mat_A, win_draws=False),
                                                      B_possible_decisions,
                                                      chunksize=16),
                                  total=len(B_possible_decisions)))

    print('Computing payoff paths for player A...')
    with mp.Pool() as pool:
        A_payoff_paths = list(tqdm(pool.imap_unordered(partial(find_paths_allocations, dest=d_A),
                                                       A_payoff_mats,
                                                       chunksize=1),
                                   total=len(A_payoff_mats)))

    A_payoff_paths = np.concatenate(A_payoff_paths)
    print('Expected payoff for player A:', A_payoff_paths.sum()/len(A_payoff_paths))


    print('Computing payoff graphs for player B...')
    with mp.Pool() as pool:
        B_payoff_mats = list(tqdm(pool.imap_unordered(partial(build_payoff_matrix, adj_mat=adj_mat_B, win_draws=False),
                                                      A_possible_decisions,
                                                      chunksize=16),
                                  total=len(A_possible_decisions)))

    print('Computing payoff paths for player B...')
    with mp.Pool() as pool:
        B_payoff_paths = list(tqdm(pool.imap_unordered(partial(find_paths_allocations, dest=d_B),
                                                       B_payoff_mats,
                                                       chunksize=1),
                                   total=len(B_payoff_mats)))

    B_payoff_paths = np.concatenate(B_payoff_paths)
    print('Expected payoff for player B:', B_payoff_paths.sum()/len(B_payoff_paths))
