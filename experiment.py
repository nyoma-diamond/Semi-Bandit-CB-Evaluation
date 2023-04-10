import glob
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import dill
import numpy as np

random.seed(42)
np.random.seed(42)

from game_data import GameData

from pdgraph import compute_bounds
from pdgraph import build_adjacency_matrix, find_paths_allocations, prune_dead_ends
from pdgraph import compute_expected_payoff, estimate_max_payoff, compute_max_possible_payoff, compute_supremum_payoff


def compute_metrics(t: int,
                    game: GameData,
                    win_draws: bool,
                    all_decisions: np.ndarray,
                    sample_threshold=None,
                    chunksize=1):
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
    pruned_graph = prune_dead_ends(bounded_graph)

    # Compute possible decisions opponent could have played
    opp_possible_decisions = find_paths_allocations(pruned_graph, game.K, opp_resources)


    with ProcessPoolExecutor() as ppe:
        # True expected payoff
        true_expected_payoff = ppe.submit(compute_expected_payoff,
                                          all_decisions, np.expand_dims(opp_decision, axis=0), win_draws,
                                          sample_threshold=sample_threshold, chunksize=chunksize)
        # Observable expected payoff
        obs_expected_payoff = ppe.submit(compute_expected_payoff,
                                         all_decisions, opp_possible_decisions, win_draws,
                                         sample_threshold=sample_threshold, chunksize=chunksize)
        # Observable max payoff
        obs_max_payoff = ppe.submit(estimate_max_payoff,
                                     opp_possible_decisions, player_resources, win_draws,
                                     sample_threshold=sample_threshold, chunksize=chunksize)
        # Supremum payoff
        sup_payoff = ppe.submit(compute_supremum_payoff,
                                opp_possible_decisions, player_resources, win_draws,
                                sample_threshold=sample_threshold, chunksize=chunksize)
        # True max payoff
        true_max_payoff = compute_max_possible_payoff(opp_decision, player_resources, win_draws)


    payoffs = np.asarray([
        true_expected_payoff.result(),
        obs_expected_payoff.result(),
        true_max_payoff,
        obs_max_payoff.result(),
        sup_payoff.result()
    ])

    regrets = payoffs - result.sum()

    return payoffs, regrets

def process_game(game: GameData,
                 A_all_decisions: np.ndarray,
                 B_all_decisions: np.ndarray,
                 sample_threshold=None,
                 chunksize=1):
    """
    Process and save data for the provided game
    NOTE: assumes the provided game is valid
    :param game: game to process
    :param A_all_decisions: All decisions available to player A
    :param B_all_decisions: All decisions available to player B
    :param sample_threshold: maximum number of operations before sampling is used to reduce the number of operations
    :param chunksize: chunksize parameter for multiprocessing
    """
    # initialize payoff and regret records
    A_payoffs, A_regrets = np.empty((0, 5)), np.empty((0, 5))
    B_payoffs, B_regrets = np.empty((0, 5)), np.empty((0, 5))

    # for each round
    for t in range(game.T):
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

    return game.identifier(), results


if __name__ == '__main__':
    # Chunk size to pass to multiprocessing
    chunksize = 64

    # Operation count threshold before sampling is used
    sample_threshold = None

    # Maximum number of games to attempt to process in parallel
    # NOTE: If this value is 1 then computation may spend a lot of time waiting for other processes to finish
    #       It's a good idea to have this be at least 2 so that another game is always ready to be processed
    # WARNING: Setting this value very high can cause system problems/crashes due to extremely large resource usage
    #          It is recommended that this value not exceed 4 as very little benefit is gained beyond that value
    max_parallel_games = 4

    # directory to load game data from
    in_dir = r'./simulations/**/*'

    # directory to save results to
    out_dir = rf'./results/{time.strftime("%Y-%m-%d_%H-%M-%S")}'

    Path(out_dir).mkdir(parents=True, exist_ok=False)
    print(f'Saving results to {out_dir}')

    games = {}

    for path in glob.glob(in_dir):
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

            with ProcessPoolExecutor(max_workers=max_parallel_games) as ppe:
                futures = [ppe.submit(process_game,
                                      game, A_all_decisions, B_all_decisions,
                                      sample_threshold=sample_threshold, chunksize=chunksize)
                           for game in games[K][(A_resources, B_resources)] if game.is_valid()]

                for future in tqdm(as_completed(futures), total=len(futures), leave=False):
                    game_id, results = future.result()
                    if results is not None:
                        filename = rf'{out_dir}/{game_id}'
                        # if a file with the desired name already exists, increment i until available
                        if Path(rf'{filename}.npy').exists():
                            i = 1
                            while Path(rf'{filename}_{i}.npy').exists():
                                i += 1
                            filename = rf'{filename}_{i}'

                        # save the data
                        np.save(filename, results)
