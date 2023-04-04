import glob
import random
from math import comb
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import dill
import numpy as np

random.seed(42)
np.random.seed(42)

from pdgraph import coordinate_to_index, compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import compute_expected_payoff, estimate_best_payoff, compute_best_possible_payoff, compute_supremum_payoff


def compute_metrics(game, win_draws, t, all_decisions, chunksize=1, track_progress=False):
    """
    Compute metrics for a player given the game and round
    :param game: game data
    :param win_draws: whether the player wins draws (False = player A, True = player B)
    :param t: the round
    :param all_decisions: the list of all possible decisions available to the player
                          NOTE: large quantities of possible decisions can cause computation to take a very long time.
                                If this happens, consider providing a uniformly sampled subset of decisions instead.
    :param chunksize: chunksize parameter for multiprocessing
    :param track_progress: whether to use tqdm to track computation progress
    :return: the payoffs and regrets observed by the player this round
    """
    if win_draws:  # Player is player B (i.e., wins draws)
        player_decision = game.B_decisions[t]
        player_resources = game.B_resources

        opp_decision = game.A_decisions[t]
        opp_resources = game.A_resources

        result = game.B_results[t]
    else:  # Player is player A (i.e., loses draws)
        player_decision = game.A_decisions[t]
        player_resources = game.A_resources

        opp_decision = game.B_decisions[t]
        opp_resources = game.B_resources

        result = game.A_results[t]

    # Compute possible opponent allocation bounds
    opp_bounds = compute_bounds(player_decision, result, opp_resources, win_draws)

    # Prune opponent's decision graph
    pruned_graph = prune_dead_ends(build_adjacency_matrix(game.K, opp_resources, opp_bounds), prune_unreachable=track_progress)

    # Compute possible decisions opponent could have played
    opp_possible_decisions = find_paths_allocations(pruned_graph, game.K, opp_resources, track_progress=track_progress)

    # Payoffs
    payoffs = np.asarray([
        # True expected payoff
        compute_expected_payoff(all_decisions, np.expand_dims(opp_decision, axis=0), win_draws, chunksize=chunksize, track_progress=track_progress),
        # Estimated expected payoff
        compute_expected_payoff(all_decisions, opp_possible_decisions, win_draws, chunksize=chunksize, track_progress=track_progress),
        # True best possible payoff
        compute_best_possible_payoff(opp_decision, player_resources, win_draws),
        # Estimated best possible payoff
        estimate_best_payoff(opp_possible_decisions, player_resources, win_draws, chunksize=chunksize, track_progress=track_progress),
        # Supremum payoff
        compute_supremum_payoff(opp_possible_decisions, player_resources, win_draws, chunksize=chunksize, track_progress=track_progress)
    ])

    regrets = payoffs - result.sum()

    return payoffs, regrets



if __name__ == '__main__':
    estimate_expected_payoff = False

    track_progress = False
    chunksize = 32

    games = {}

    for path in glob.glob(r'./simulations/**/*'):
        with open(path, 'rb') as  f:
            game = dill.load(f)

            if game.K not in games.keys():
                games[game.K] = {(game.A_resources, game.B_resources): [game]}
            elif (game.A_resources, game.B_resources) not in games[game.K].keys():
                games[game.K][(game.A_resources, game.B_resources)] = [game]
            else:
                games[game.K][(game.A_resources, game.B_resources)].append(game)

    for K in games.keys():
        for (A_resources, B_resources) in games[K].keys():
            A_graph = build_adjacency_matrix(K, A_resources)
            B_graph = build_adjacency_matrix(K, B_resources)

            A_num_decisions = comb(K+A_resources-1, K-1)
            B_num_decisions = comb(K+B_resources-1, K-1)

            A_all_decisions = find_paths_allocations(A_graph, K, A_resources, track_progress=track_progress)
            B_all_decisions = find_paths_allocations(B_graph, K, B_resources, track_progress=track_progress)

            d_A = coordinate_to_index((K, A_resources), K, A_resources)
            d_B = coordinate_to_index((K, B_resources), K, B_resources)

            for game in games[K][(A_resources, B_resources)]:
                for t in range(game.T):
                    # ========== PLAYER A METRICS ==========
                    A_payoffs, A_regrets = compute_metrics(game, False, t, A_all_decisions, chunksize=chunksize, track_progress=track_progress)
                    print(game.A_decisions[t])
                    print(A_payoffs)
                    print(A_regrets)

                    # ========== PLAYER B METRICS ==========
                    B_payoffs, B_regrets = compute_metrics(game, True, t, B_all_decisions, chunksize=chunksize, track_progress=track_progress)
                    print(game.B_decisions[t])
                    print(B_payoffs)
                    print(B_regrets)

