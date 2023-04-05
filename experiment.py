import glob
import random
import time
from pathlib import Path

from tqdm import tqdm
import dill
import numpy as np

random.seed(42)
np.random.seed(42)

from game_data import GameData

from pdgraph import compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import compute_expected_payoff, estimate_best_payoff, compute_best_possible_payoff, compute_supremum_payoff


def compute_metrics(t: int,
                    game: GameData,
                    win_draws: bool,
                    all_decisions: np.ndarray,
                    sample_threshold=None,
                    chunksize=1,
                    track_progress=False):
    """
    Compute metrics for a player given the game and round
    :param t: the round
    :param game: game data
    :param win_draws: whether the player wins draws (False = player A, True = player B)
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
    chunksize = 64
    sample_threshold = None

    out_dir = rf'./results/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    Path(out_dir).mkdir(parents=True, exist_ok=False)

    print(f'Saving results to {out_dir}')

    games = {}

    # Load game data (**/* LOADS ALL FILES IN ALL SUBDIRECTORIES OF ./simulations)
    for path in glob.glob(r'./simulations/**/*'):
        with open(path, 'rb') as f:
            game = dill.load(f)

            if game.K not in games.keys():
                games[game.K] = {(game.A_resources, game.B_resources): [game]}
            elif (game.A_resources, game.B_resources) not in games[game.K].keys():
                games[game.K][(game.A_resources, game.B_resources)] = [game]
            else:
                games[game.K][(game.A_resources, game.B_resources)].append(game)

    # for each tested number of battlefields
    for K in tqdm(games.keys(), leave=True):
        # for each resource matchup
        for (A_resources, B_resources) in tqdm(games[K].keys(), leave=False):
            # build decision graphs for players
            A_graph = build_adjacency_matrix(K, A_resources)
            B_graph = build_adjacency_matrix(K, B_resources)

            # generate all possible decisions
            A_all_decisions = find_paths_allocations(A_graph, K, A_resources)
            B_all_decisions = find_paths_allocations(B_graph, K, B_resources)

            # for each game (algorithm matchup)
            for game in tqdm(games[K][(A_resources, B_resources)], leave=False):
                # something went wrong in the match and there's no data; skip
                if not game.is_valid():
                    continue

                # initialize payoff and regret records
                A_payoffs, A_regrets = np.empty((0, 5)), np.empty((0, 5))
                B_payoffs, B_regrets = np.empty((0, 5)), np.empty((0, 5))

                # for each round
                for t in tqdm(range(game.T), leave=False):
                    # compute payoff & regret metrics for player A
                    A_round_payoff, A_round_regret = compute_metrics(t, game, False, A_all_decisions, sample_threshold=sample_threshold, chunksize=chunksize)
                    A_payoffs = np.vstack((A_payoffs, A_round_payoff))
                    A_regrets = np.vstack((A_regrets, A_round_regret))

                    # compute payoff & regret metrics for player B
                    B_round_payoff, B_round_regret = compute_metrics(t, game, True, B_all_decisions, sample_threshold=sample_threshold, chunksize=chunksize)
                    B_payoffs = np.vstack((B_payoffs, B_round_payoff))
                    B_regrets = np.vstack((B_regrets, B_round_regret))

                # prepare data for saving to file
                results = np.stack(((A_payoffs, A_regrets), (B_payoffs, B_regrets)))

                filename = rf'{out_dir}/{game.identifier()}'
                # if a file with the desired name already exists, increment i until available
                if Path(rf'{filename}.npy').exists():
                    i = 1
                    while Path(rf'{filename}_{i}.npy').exists():
                        i += 1
                    filename = rf'{filename}_{i}'

                # save the data
                np.save(filename, results)