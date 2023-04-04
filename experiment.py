import glob
import random
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from tqdm import tqdm
import dill
import numpy as np

random.seed(42)
np.random.seed(42)

from game_data import GameData

from pdgraph import compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import compute_expected_payoff, estimate_best_payoff, compute_best_possible_payoff, compute_supremum_payoff


def compute_metrics(game: GameData,
                    win_draws: bool,
                    t: int,
                    all_decisions: np.ndarray,
                    sample_threshold=None,
                    chunksize=1,
                    track_progress=False):
    """
    Compute metrics for a player given the game and round
    :param game: game data
    :param win_draws: whether the player wins draws (False = player A, True = player B)
    :param t: the round
    :param all_decisions: the list of all possible decisions available to the player
                    NOTE: large quantities of possible decisions can cause computation to take a very long time.
                          If this happens, consider providing a uniformly sampled subset of decisions instead.
    :param sample_threshold: maximum number of operations before sampling is used to reduce the number of operations
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

    # Generate opponent's decision graph considering allocation bounds
    bounded_graph = build_adjacency_matrix(game.K, opp_resources, opp_bounds)

    # Prune nodes that lead to dead-ends
    pruned_graph = prune_dead_ends(bounded_graph, prune_unreachable=track_progress)

    # Compute possible decisions opponent could have played
    opp_possible_decisions = find_paths_allocations(pruned_graph, game.K, opp_resources, track_progress=track_progress)


    payoffs = np.asarray([
        # True expected payoff
        compute_expected_payoff(all_decisions, np.expand_dims(opp_decision, axis=0), win_draws,
                                sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress),
        # Estimated expected payoff
        compute_expected_payoff(all_decisions, opp_possible_decisions, win_draws,
                                sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress),
        # True best possible payoff
        compute_best_possible_payoff(opp_decision, player_resources, win_draws),
        # Estimated best possible payoff
        estimate_best_payoff(opp_possible_decisions, player_resources, win_draws,
                             sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress),
        # Supremum payoff
        compute_supremum_payoff(opp_possible_decisions, player_resources, win_draws,
                                sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress)
    ])

    regrets = payoffs - result.sum()

    return payoffs, regrets



if __name__ == '__main__':
    track_progress = False
    chunksize = 100
    sample_threshold = None

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
        for (A_resources, B_resources) in tqdm(games[K].keys(), leave=True):

            A_graph = build_adjacency_matrix(K, A_resources)
            B_graph = build_adjacency_matrix(K, B_resources)

            A_all_decisions = find_paths_allocations(A_graph, K, A_resources, track_progress=track_progress)
            B_all_decisions = find_paths_allocations(B_graph, K, B_resources, track_progress=track_progress)

            for game in tqdm(games[K][(A_resources, B_resources)], leave=False):
                for t in tqdm(range(game.T), leave=False):
                    # ========== PLAYER A METRICS ==========
                    A_payoffs, A_regrets = compute_metrics(game, False, t, A_all_decisions, sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress)

                    # ========== PLAYER B METRICS ==========
                    B_payoffs, B_regrets = compute_metrics(game, True, t, B_all_decisions, sample_threshold=sample_threshold, chunksize=chunksize, track_progress=track_progress)

