from math import comb
from functools import partial
import multiprocessing as mp

from tqdm import tqdm
import numpy as np


def coordinate_to_index(coordinate: tuple[int, int], battlefields: int, N: int) -> int:
    """
    Converts a provided coordinate to the relevant adjacency matrix index
    :param coordinate: coordinate to convert
    :param battlefields: number of battlefields
    :param N: resources available to the player
    :return: the equivalent index. -1 if invalid coordinate
    """
    if coordinate[0] == coordinate[1] == 0:
        return 0
    elif coordinate[0] == battlefields and coordinate[1] == N:
        return (battlefields - 1) * (N + 1) + 1
    elif coordinate[0] <= 0 or coordinate[0] >= battlefields or coordinate[1] < 0:
        return -1
    else:
        return (coordinate[0] - 1) * (N + 1) + coordinate[1] + 1


def index_to_coordinate(index: int, battlefields: int, N: int) -> tuple[int, int]:
    """
    Converts a provided adjacency matrix index to relevant graph coordinate
    :param index: index to convert
    :param battlefields: the number of battlefields
    :param N: resources available to the player
    :return: the equivalent coordinate. (-1,-1) if invalid index is provided
    """
    if index < 0 or index > (battlefields - 1) * (N + 1) + 1:
        return -1, -1
    if index == 0:
        return 0, 0
    if index == (battlefields - 1) * (N + 1) + 1:
        return battlefields, N
    return (index + N) // (N + 1), (index - 1) % (N + 1)


def get_child_indices(node: int, battlefields: int, N: int) -> tuple[int, int]:
    """
    get the indices of the children of the provided node
    :param node: node to get children of
    :param battlefields: number of battlefields
    :param N: number of resources
    :return: index of first child, 1+index of last child
    """
    node_coord = index_to_coordinate(node, battlefields, N)

    children_end = coordinate_to_index((node_coord[0] + 1, N), battlefields, N) + 1

    if node_coord[0] < battlefields - 1:
        children_start = coordinate_to_index((node_coord[0] + 1, node_coord[1]), battlefields, N)
    else:
        children_start = (battlefields - 1) * (N + 1) + 1

    return children_start, children_end


def build_adjacency_matrix(battlefields: int,
                           N: int,
                           bounds: np.ndarray = None) -> np.ndarray:
    """
    Creates adjacency matrix for possible allocations/decisions for a player
    :param battlefields: the number of battlefields in the game
    :param N: the number of resources available to the player
    :param bounds: lower and upper bounds for possible allocations
    :return: Adjacency matrix for graph representing possible allocations/decisions of style mat[from][to]
    """
    adj_mat = np.full(((battlefields - 1) * (N + 1) + 2, (battlefields - 1) * (N + 1) + 2), -1, dtype=int)

    if bounds is None:
        bounds = np.full((battlefields, 2), (0, N), dtype=np.ubyte)

    for frm_i in range(adj_mat.shape[0] - 1):  # can skip last node because it's a dead end
        to_start, to_end = get_child_indices(frm_i, battlefields, N)

        if to_end == adj_mat.shape[1]:
            allocations = np.asarray([adj_mat.shape[0] - frm_i - 2])
        else:
            allocations = np.arange(to_end - to_start)

        frm_coord = index_to_coordinate(frm_i, battlefields, N)

        bound = bounds[frm_coord[0]]
        allocations[(bound[0] > allocations) | (allocations > bound[1])] = -1

        adj_mat[frm_i, to_start:to_end] = allocations

    return adj_mat


def prune_dead_ends(adj_mat: np.ndarray, prune_unreachable: bool = False) -> np.ndarray:
    """
    Removes outgoing edges from all nodes that cannot reach the final node
    :param adj_mat: adjacency matrix representing the graph to prune
    :param prune_unreachable: whether to also remove unreachable nodes
    :return: the pruned adjacency matrix
    """
    adj_mat = adj_mat.copy()

    for i in range(adj_mat.shape[0] - 2, -1, -1):  # need -2 instead of -1 because final node never has any children
        if (adj_mat[i] == -1).all():
            adj_mat[:, i] = -1

    if prune_unreachable:
        for i in range(1, adj_mat.shape[0] - 1):  # need to start at 1 because the first node has no parents
            if (adj_mat[:, i] == -1).all():
                adj_mat[i] = -1

    return adj_mat


def find_subpaths_subworker(prev, adj_mat, *args, **kwargs):
    """
    Sub-worker function for pathfinding
    :param prev: node being exited
    :param adj_mat: adjacency matrix representing the DAG being searched
    :param args: positional arguments to pass to find_subpaths_allocations
    :param kwargs: keyword arguments to pass to find_subpaths_allocations
    :return: subpaths out of the node in *args, with the previous node appended
    """
    subpaths, child = find_subpaths_allocations(adj_mat, *args, **kwargs)
    prepend = np.full((len(subpaths), 1), adj_mat[prev, child], dtype=np.ubyte)
    return np.concatenate((prepend, subpaths), axis=1)


def find_subpaths_allocations(adj_mat: np.ndarray,
                              node: int,
                              dest: int,
                              visited: dict[int, np.ndarray],
                              battlefields: int,
                              N: int,
                              pbar: tqdm = None) -> tuple[np.ndarray, int]:
    """
    Find all paths (partial allocations) between the provided nodes in the provided DAG, represented by their corresponding resource allocations
    :param node: the starting node
    :param dest: the destination node
    :param visited: dictionary of all subpaths from nodes already visited (partial allocations)
    :param adj_mat: adjacency matrix representing the DAG being searched
    :param battlefields: the number of battlefields
    :param N: the number of resources available to the player
    :param pbar: tqdm progress bar for progress tracking
    :return: all paths (partial allocations) between the provided nodes, and the starting node
    """
    if node == dest:
        return np.empty(shape=(1, 0), dtype=np.ubyte), node
    elif node not in visited.keys():
        children_start, children_end = get_child_indices(node, battlefields, N)

        visited[node] = np.vstack([find_subpaths_subworker(node, adj_mat, child, dest, visited, battlefields, N, pbar)
                                   for child in range(children_start, children_end)
                                   if adj_mat[node, child] != -1])

        if pbar is not None:
            pbar.update()

    return visited[node], node


def find_paths_allocations(adj_mat: np.ndarray,
                           battlefields: int,
                           N: int,
                           track_progress: bool = False) -> np.ndarray:
    """
    Find all possible paths (decisions) through the provided DAG
    :param adj_mat: adjacency matrix representing the DAG being searched
    :param battlefields: the number of battlefields
    :param N: the number of resources available to the player
    :param track_progress: whether to use tqdm to track progress (estimated by proportion of nodes fully explored)
                     NOTE: estimate assumes all nodes are accessible from the source node
    :return: all possible allocations (paths through the DAG)
    """
    pbar = None
    if track_progress:
        nodes = (adj_mat != -1).any(axis=1).sum()
        pbar = tqdm(total=nodes)

    paths, _ = find_subpaths_allocations(adj_mat, 0, len(adj_mat)-1, {}, battlefields, N, pbar)

    if track_progress:
        pbar.close()

    return paths


def allocation_by_id(id: int, battlefields: int, N: int) -> np.ndarray:
    """
    Computes the allocation decision associated with the provided lexicographical index
    :param id: id to compute decision from
    :param battlefields: number of battlefields for allocation
    :param N: resources available to allocate
    :return: the allocations as a list of ints
    """
    if battlefields == 1:
        return np.asarray([N], dtype=np.ubyte)

    N_ = N
    unit = comb(battlefields + N - 2, battlefields - 2)
    while unit <= id:
        id -= unit
        N_ -= 1
        unit = comb(battlefields + N_ - 2, battlefields - 2)

    return np.append(N-N_, allocation_by_id(id, battlefields - 1, N_)).astype(np.ubyte)


def compute_expected_payoff_for_decision(decision: np.ndarray,
                                         opp_decisions: np.ndarray,
                                         win_draws: bool) -> float:
    """
    Compute the expected payoff for a given decision
    :param decision: decision to compute the expected payoff of
    :param opp_decisions: all possible decisions the opponent can take
    :param win_draws: whether the player wins draws
    :return: the expected payoff (total available payoff if divide is False)
    """
    compare = np.greater_equal if win_draws else np.greater
    total = sum(compare(decision, opp_dec).sum() for opp_dec in opp_decisions)
    return total / len(opp_decisions)


def compute_expected_payoff(target_decisions: np.ndarray,
                            opp_decisions: np.ndarray,
                            win_draws: bool,
                            sample_threshold: int = None,
                            chunksize: int = 1,
                            track_progress: bool = False) -> float:
    """
    Compute the expected payoff for a set of decisions
    :param target_decisions: set of decisions to compute the expected payoff of
    :param opp_decisions: all possible decisions the opponent can take
    :param win_draws: whether the player wins draws
    :param sample_threshold: maximum number of operations before sampling is used to reduce the number of operations
                       NOTE: to bring down the number of operations, whichever set has a larger sample size will have
                             its sample size divided by 10 until the number of operations is less than the threshold
    :param chunksize: chunksize parameter for multiprocessing
    :param track_progress: whether to use tqdm to track computation progress
    :return: expected payoff of the provided set of decisions
    """
    if sample_threshold is not None and len(target_decisions)*len(opp_decisions) > sample_threshold:
        num_samples = np.asarray([len(target_decisions), len(opp_decisions)], dtype=np.float_)

        while num_samples.prod() > sample_threshold:
            num_samples[num_samples.argmax()] //= 10

        num_samples = num_samples.astype(np.int_)

        if num_samples[0] != len(target_decisions):
            target_decisions = target_decisions[np.random.randint(len(target_decisions), size=num_samples[0])]

        if num_samples[1] != len(opp_decisions):
            opp_decisions = opp_decisions[np.random.randint(len(opp_decisions), size=num_samples[1])]


    expected_payoff = 0

    if track_progress:
        pbar = tqdm(total=len(target_decisions) * len(opp_decisions), unit_scale=True)

    with mp.Pool() as pool:
        for i, partial_payoff in enumerate(pool.imap_unordered(partial(compute_expected_payoff_for_decision,
                                                                       opp_decisions=opp_decisions,
                                                                       win_draws=win_draws),
                                                               target_decisions,
                                                               chunksize=chunksize)):
            expected_payoff += partial_payoff
            if track_progress:
                pbar.update(len(opp_decisions))
                pbar.set_postfix_str(f'Expected payoff: {expected_payoff / (i + 1)}')

    expected_payoff /= len(target_decisions)

    return expected_payoff


def compute_max_possible_payoff(opp_decision: np.ndarray, N: int, win_draws: bool) -> int:
    """
    Computes the max possible payoff against the provided decision
    :param opp_decision: decision by opponent
    :param N: available resources
    :param win_draws: whether the player wins draws or not
    :return: the max possible payoff
    """
    dec = sorted(opp_decision, reverse=True)

    payoff = 0

    while len(dec) > 0:
        cur = dec.pop()
        if cur < N + win_draws:
            payoff += 1
            N -= cur + (not win_draws)
        else:
            break

    return payoff


def estimate_max_payoff(opp_decisions: np.ndarray,
                         N: int,
                         win_draws: bool,
                         sample_threshold: int = None,
                         chunksize: int = 1,
                         track_progress: bool = False):
    """
    Computes the expected value for the max payoff (observable max payoff)
    :param opp_decisions: set of decisions possible to be played by the opponent
    :param N: resources available to the player
    :param win_draws: whether the player wins draws or not
    :param sample_threshold: maximum number of decisions before sampling is used to reduce the number of operations
    :param chunksize: chunksize parameter for multiprocessing
    :param track_progress: whether to use tqdm to track computation progress
    :return: expected value for the max possible payoff
    """
    if sample_threshold is not None and len(opp_decisions) > sample_threshold:
        opp_decisions = opp_decisions[np.random.randint(len(opp_decisions), size=sample_threshold)]


    with mp.Pool() as pool:
        max_payoffs = pool.imap_unordered(partial(compute_max_possible_payoff, N=N, win_draws=win_draws),
                                           opp_decisions,
                                           chunksize=chunksize)
        if track_progress:
            max_payoffs = tqdm(max_payoffs,
                                total=len(opp_decisions),
                                unit_scale=True,
                                miniters=chunksize*100,
                                mininterval=0.2)


        total_payoff = sum(max_payoffs)

    return total_payoff / len(opp_decisions)


def compute_supremum_payoff(opp_decisions: np.ndarray,
                            N: int,
                            win_draws: bool,
                            sample_threshold: int = None,
                            chunksize: int = 1,
                            track_progress: bool = False):
    """
    Computes the supremum possible payoff (i.e., the minimum max possible value for payoff)
    :param opp_decisions: set of decisions possible to be played by the opponent
    :param N: resources available to the player
    :param win_draws: whether the player wins draws or not
    :param sample_threshold: maximum number of decisions before sampling is used to reduce the number of operations
    :param chunksize: chunksize parameter for multiprocessing
    :param track_progress: whether to use tqdm to track computation progress
    :return: supremum payoff
    """
    if sample_threshold is not None and len(opp_decisions) > sample_threshold:
        opp_decisions = opp_decisions[np.random.randint(len(opp_decisions), size=sample_threshold)]

    with mp.Pool() as pool:
        max_payoffs = pool.imap_unordered(partial(compute_max_possible_payoff, N=N, win_draws=win_draws),
                                           opp_decisions,
                                           chunksize=chunksize)
        if track_progress:
            max_payoffs = tqdm(max_payoffs,
                                total=len(opp_decisions),
                                unit_scale=True,
                                miniters=chunksize*100,
                                mininterval=0.2)


        supremum_payoff = min(max_payoffs)

    return supremum_payoff


def make_discrete_allocation(allocation: np.ndarray, N: int):
    """
    Converts a set of continuous allocations to discrete allocations
    NOTE: remainders are treated as proportional probability of having the "partial" resource
    :param allocation: continuous allocations to convert (assumed to sum to 1)
    :param N: total resources available
    :return: discrete allocations
    """
    allocation *= N
    discrete = np.floor(allocation).astype(np.ubyte)
    rem = np.mod(allocation, 1)

    if N - sum(discrete) > 0:  # Allocate extra resources randomly with probability proportional to remainder
        flips = np.random.choice(range(rem.size),
                                 size=int(N - sum(discrete)),
                                 replace=False,
                                 p=rem/sum(rem))
        discrete[flips] += 1

    return discrete


def compute_bounds(decision: np.ndarray,
                   result: np.ndarray,
                   opp_resources: int,
                   win_draws: bool) -> np.ndarray:
    """
    Compute the bounds on the opponent's possible allocations
    :param decision: player's allocations
    :param result: vector of battlefield wins
    :param opp_resources: resources believed available to the opponent
    :param win_draws: whether the relevant player wins draws or not
    :return: array containing the bounds of the opponent's possible allocations
    """
    lb = np.where(result, 0, decision + win_draws)

    ub = np.where(result,
                  decision - (not win_draws),
                  opp_resources + (1 - sum(~result)) * win_draws - (sum(decision[~result]) - decision))

    return np.column_stack((lb, ub))
