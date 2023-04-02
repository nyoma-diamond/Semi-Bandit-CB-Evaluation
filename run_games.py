from pathlib import Path
import random
import time
from functools import partial
from itertools import combinations_with_replacement, product
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import dill

from algorithms.cb_algorithm import CB_Algorithm

random.seed(42)
np.random.seed(42)

from algorithms.cucb_dra import CUCB_DRA
from algorithms.edge import Edge
from algorithms.mara import MARA
from algorithms.random_algorithm import Random_Allocation

from game_data import GameData


def play_game(player_A: CB_Algorithm,
              player_B: CB_Algorithm,
              K: int,
              T: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Play a full game
    :param player_A: The model for player A
    :param player_B: The model for player A
    :param K: The number of battlefields (assumes players were initialized with this value)
    :param T: The number of rounds to play
    :return: The players' decisions and their associated per-battlefield results (wins/losses) during the game
    """
    # Initialize empty decision and result arrays
    A_decisions, A_results = np.empty(shape=(0, K), dtype=np.bool_), np.empty(shape=(0, K), dtype=np.bool_)
    B_decisions, B_results = np.empty(shape=(0, K), dtype=np.bool_), np.empty(shape=(0, K), dtype=np.bool_)

    # Play the game for T rounds
    for _ in range(T):
        # Both players generate a decision for the round
        A_decision = player_A.generate_decision()
        B_decision = player_B.generate_decision()

        # Insert the decisions into the arrays of previous decisions
        A_decisions = np.vstack((A_decisions, A_decision))
        B_decisions = np.vstack((B_decisions, B_decision))

        # Compute results of the round
        A_result = np.greater(A_decision, B_decision)
        B_result = ~A_result

        # Insert the results into the arrays of previous results
        A_results = np.vstack((A_results, A_result))
        B_results = np.vstack((B_results, B_result))

        # Update the players based on their results
        player_A.update(A_result)
        player_B.update(B_result)

    return (A_decisions, A_results), (B_decisions, B_results)


def game_worker(args: tuple, T: int) -> GameData:
    """
    Worker function for running a game simulation and reporting the results
    :param args: arguments in the form of (((A algorithm, arguments), (B algorithm, arguments)), (A resources, B resources), Battlefields)
    :param T: The number of rounds to play
    :return: A GameData object containing the parameters of the game, the players' decisions, and their results
    """
    ((A_alg, A_kwargs), (B_alg, B_kwargs)), (A_resources, B_resources), K = args  # extract arguments

    # Initialize the players
    player_A = A_alg(K, A_resources, **A_kwargs)
    player_B = B_alg(K, B_resources, **B_kwargs)

    # Play the game
    A_game, B_game = play_game(player_A, player_B, K, T)

    return GameData(K, T, A_alg.__name__, A_resources, *A_game, B_alg.__name__, B_resources, *B_game)


if __name__ == '__main__':
    battlefields = [5]
    resources = [10, 15, 20]
    T = 100

    algorithms = {
        MARA: dict(c=2.5),
        CUCB_DRA: dict(),
        Edge: dict(gamma=0.25),
        Random_Allocation: dict()
    }

    out_dir = rf'./simulations/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    params = list(product(product(algorithms.items(), repeat=2),
                     combinations_with_replacement(sorted(resources, reverse=True), 2),
                     battlefields))

    with mp.Pool() as pool:
        for game in tqdm(pool.imap_unordered(partial(game_worker, T=T), params),
                                                     total=len(params)):
            filename = '-'.join(str(x) for x in [game.T, game.K, game.A_algorithm, game.A_resources, game.B_algorithm, game.B_resources])
            with open(rf'{out_dir}/{filename}', 'wb') as file:
                dill.dump(game, file)
