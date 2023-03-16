import numpy as np
from math import comb
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from time import sleep

def coordinate_to_index(coordinate: tuple[int,int], battlefields: int, N: int) -> int:
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
        return (battlefields-1) * (N+1) + 1
    elif coordinate[0] <= 0 or coordinate[0] >= battlefields or coordinate[1] < 0:
        return -1
    else:
        return (coordinate[0]-1) * (N+1) + coordinate[1] + 1


def index_to_coordinate(index: int, battlefields: int, N: int) -> tuple[int,int]:
    """
    Converts a provided adjacency matrix index to relevant graph coordinate
    :param index: index to convert
    :param battlefields: the number of battlefields
    :param N: resources available to the player
    :return: the equivalent coordinate. (-1,-1) if invalid index is provided
    """
    if index < 0 or index > (battlefields-1) * (N+1) + 1:
        return -1, -1
    if index == 0:
        return 0, 0
    if index == (battlefields-1) * (N+1) + 1:
        return battlefields, N
    return (index+N) // (N+1), (index-1) % (N+1)


def get_child_indices(node: int, battlefields: int, N: int) -> tuple[int,int]:
    """
    get the indices of the children of the provided node
    :param node: node to get children of
    :param battlefields: number of battlefields
    :param N: number of resources
    :return: index of first child, 1+index of last child
    """
    node_coord = index_to_coordinate(node, battlefields, N)

    children_end = coordinate_to_index((node_coord[0]+1, N), battlefields, N) + 1

    if node_coord[0] < battlefields-1:
        children_start = coordinate_to_index((node_coord[0]+1, node_coord[1]), battlefields, N)
    else:
        children_start = (battlefields-1) * (N+1) + 1

    return children_start, children_end



def build_adjacency_matrix(battlefields: int,
                           N: int,
                           bounds: list[tuple[int,int]] = None) -> np.ndarray:
    """
    Creates adjacency matrix for possible allocations/decisions for a player
    :param battlefields: the number of battlefields in the game
    :param N: the number of resources available to the player
    :param bounds: list of lower and upper bounds for possible allocations
    :return: Adjacency matrix for graph representing possible allocations/decisions of style mat[from][to]
    """
    adj_mat = np.full(((battlefields-1) * (N+1) + 2, (battlefields-1) * (N+1) + 2), -1, dtype=int)

    if bounds is None:
        bounds = [(0,N)]*battlefields


    for frm_i in range(adj_mat.shape[0]-1): # can skip last node because it's a dead end
        to_start, to_end = get_child_indices(frm_i, battlefields, N)

        if to_end == adj_mat.shape[1]:
            allocations = np.asarray([adj_mat.shape[0]-frm_i-2])
        else:
            allocations = np.arange(to_end-to_start)

        frm_coord = index_to_coordinate(frm_i, battlefields, N)

        bound = bounds[frm_coord[0]]
        allocations[(bound[0] > allocations) | (allocations > bound[1])] = -1

        adj_mat[frm_i, to_start:to_end] = allocations

    return adj_mat


def prune_dead_ends(adj_mat: np.ndarray, prune_unreachable=False) -> np.ndarray:
    """
    Removes outgoing edges from all nodes that cannot reach the final node
    :param adj_mat: adjacency matrix representing the graph to prune
    :param prune_unreachable: whether to also remove unreachable nodes
    :return: the pruned adjacency matrix
    """
    adj_mat = adj_mat.copy()

    for i in range(adj_mat.shape[0]-2, -1, -1): # need -2 instead of -1 because final node never has any children
        if (adj_mat[i]==-1).all():
            adj_mat[:,i] = -1

    if prune_unreachable:
        for i in range(1, adj_mat.shape[0]-1): # need to start at 1 because the first node has no parents
            if (adj_mat[:,i]==-1).all():
                adj_mat[i] = -1

    return adj_mat


def find_subpaths_allocations(adj_mat: np.ndarray,
                              node: int,
                              dest: int,
                              visited: dict[int, list[list[int]]],
                              battlefields: int,
                              N: int,
                              pbar: tqdm = None) -> list[list[int]]:
    """
    Find all paths (partial allocations) between the provided nodes in the provided DAG, represented by their corresponding resource allocations
    :param node: the starting node
    :param dest: the destination node
    :param visited: dictionary of all subpaths from nodes already visited (partial allocations)
    :param adj_mat: adjacency matrix representing the DAG being searched
    :param battlefields: the number of battlefields
    :param N: the number of resources available to the player
    :param pbar: tqdm progress bar for progress tracking
    :return: all paths (partial allocations) between the provided nodes
    """
    if node == dest:
        return [[]]
    elif node not in visited.keys():
        children_start, children_end = get_child_indices(node, battlefields, N)

        visited[node] = [[adj_mat[node, child]] + subpath
                         for child in range(children_start, children_end)
                         if adj_mat[node, child] != -1
                         for subpath in find_subpaths_allocations(adj_mat, child, dest, visited, battlefields, N, pbar)]

        if pbar is not None:
            pbar.update()

    return visited[node]


def find_paths_allocations(adj_mat: np.ndarray,
                           dest: int,
                           battlefields: int,
                           N: int,
                           track_progress=False) -> list[list[int]]:
    """
    Find all possible paths (decisions) through the provided DAG
    :param dest: the destination node
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

    paths = find_subpaths_allocations(adj_mat, 0, dest, {}, battlefields, N, pbar)

    if track_progress:
        pbar.close()
        sleep(0.1) # fudge to make sure printouts don't get messed up

    return paths


# NOTE: this is actually slower than graph search!
def build_allocations(battlefields: int, N: int) -> list[list[int]]:
    """
    Build the list of all possible allocations
    :param battlefields: battlefields for allocation
    :param N: resources available to allocate
    :return: all possible allocations
    """
    if battlefields == 1:
        return [[N]]

    return [[n] + alloc
            for n in range(N+1)
            for alloc in build_allocations(battlefields-1, N-n)]


def allocation_by_id(id: int, battlefields: int, N: int) -> list[int]:
    """
    Computes the allocation decision associated with the provided lexicographical index
    :param id: id to compute decision from
    :param battlefields: number of battlefields for allocation
    :param N: resources available to allocate
    :return: the allocations as a list of ints
    """
    if battlefields == 1:
        return [N]

    i = 0
    offset = 0
    unit = comb(battlefields+N-2-i, battlefields-2)
    while offset + unit <= id:
        offset += unit
        i += 1
        unit = comb(battlefields+N-2-i, battlefields-2)

    return [i] + allocation_by_id(id-offset, battlefields-1, N-i)


def compute_expected_payoff_for_decision(decision: list[int],
                                         opp_decisions: list[list[int]],
                                         win_draws=False) -> float:
    """
    Compute the expected payoff for a given decision
    :param decision: decision to compute the expected payoff of
    :param opp_decisions: all possible decisions the opponent can take
    :param win_draws: whether the player wins draws
    :return: the expected payoff (total available payoff if divide is False)
    """
    compare = np.greater_equal if win_draws else np.greater
    total = sum(compare(dec, decision).sum() for dec in opp_decisions)
    return total/len(opp_decisions)


def compute_expected_payoff(target_decisions: list[list[int]],
                            opp_decisions: list[list[int]],
                            win_draws=False,
                            chunksize=1) -> float:
    """
    Compute the expected payoff for a set of decisions
    :param target_decisions: set of decisions to compute the expected payoff of
    :param opp_decisions: all possible decisions the opponent can take
    :param win_draws: whether the player wins draws
    :param chunksize: chunksize parameter passed to pool.imap_unordered
    :return: expected payoff of the provided set of decisions
    """
    expected_payoff = 0

    with tqdm(total=len(target_decisions)*len(opp_decisions), unit_scale=True) as pbar:
        with mp.Pool() as pool:
            for i, partial_payoff in enumerate(pool.imap_unordered(partial(compute_expected_payoff_for_decision,
                                                                           opp_decisions=opp_decisions,
                                                                           win_draws=win_draws),
                                                                   target_decisions,
                                                                   chunksize=chunksize)):
                expected_payoff += partial_payoff
                pbar.update(len(opp_decisions))
                pbar.set_postfix_str(f'Expected payoff: {expected_payoff / (i+1)}')

    sleep(0.1) # fudge to make sure printouts don't get messed up

    expected_payoff /= len(target_decisions)

    return expected_payoff


def best_possible_payoff(opp_decision: list[int], N: int, win_draws=False) -> int:
    """
    Computes the best possible payoff against the provided decision
    :param opp_decision: decision by opponent
    :param N: available resources
    :param win_draws: whether the player wins draws or not
    :return: the best possible payoff
    """
    dec = sorted(opp_decision, reverse=True)

    payoff = 0

    while len(dec) > 0:
        cur = dec.pop()
        if cur >= N + win_draws:
            break
        else:
            payoff += 1
            N -= cur + (not win_draws)

    return payoff


def compute_expected_best_payoff(opp_decisions: list[list[int]], N: int, win_draws=False):
    """
    Computes the expected value for the best possible payoff
    :param opp_decisions: set of decisions possible to be played by the opponent
    :param N: resources available to the player
    :param win_draws: whether the player wins draws or not
    :return: expected value for the best possible payoff
    """
    use_tqdm = False
    if len(opp_decisions) > 1e5:
        use_tqdm = True
        opp_decisions = tqdm(opp_decisions, unit_scale=True, miniters=len(opp_decisions)/1e4, mininterval=0.2, leave=False)
    total_payoff = sum(best_possible_payoff(decision, N=N, win_draws=win_draws) for decision in opp_decisions)
    if use_tqdm:
        opp_decisions.close()
    return total_payoff / len(opp_decisions)