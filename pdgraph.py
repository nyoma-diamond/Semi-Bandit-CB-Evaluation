import numpy as np
from math import comb

def coordinate_to_index(coordinate: tuple[int,int], battlefields: int, N: int) -> int:
    """
    Converts a provided coordinate to the relevant adjacency matrix index
    :param coordinate: coordinate to convert
    :param battlefields: number of battlefields
    :param N: N (total resources) for relevant player
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
    :param N: N (total resources) for relevant player
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



def build_adjacency_matrix(battlefields: int, N: int, bounds: list[tuple[int,int]] = None) -> np.ndarray:
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


def prune_dead_ends(adj_mat: np.ndarray):
    """
    Removes outgoing edges from all nodes that cannot reach the final node
    :param adj_mat: adjacency matrix representing the graph to prune
    :return: the pruned adjacency matrix
    """
    adj_mat = adj_mat.copy()

    for i in range(adj_mat.shape[0]-2, -1, -1): # need -2 instead of -1 because final node never has any children
        if (adj_mat[i]==-1).all():
            adj_mat[:,i] = -1

    return adj_mat





def find_subpaths_allocations(adj_mat: np.ndarray, node: int, dest: int,
                              visited: dict[int, list[list[int]]], battlefields: int, N: int):
    """
    Find all paths (partial allocations) between the provided nodes in the provided DAG, represented by their corresponding resource allocations
    :param node: the starting node
    :param dest: the destination node
    :param visited: dictionary of all subpaths from nodes already visited (partial allocations)
    :param adj_mat: adjacency matrix representing the DAG being searched
    :return: all paths (partial allocations) between the provided nodes
    """
    if node == dest:
        return [[]]
    elif node not in visited.keys():
        children_start, children_end = get_child_indices(node, battlefields, N)

        visited[node] = [[adj_mat[node, child]] + subpath
                         for child in range(children_start, children_end)
                         if adj_mat[node, child] != -1
                         for subpath in find_subpaths_allocations(adj_mat, child, dest, visited, battlefields, N)]

    return visited[node]


def find_paths_allocations(adj_mat: np.ndarray, dest: int, battlefields: int, N: int):
    """
    Find all possible paths (decisions) through the provided DAG
    :param dest: the destination node
    :param adj_mat: adjacency matrix representing the DAG being searched
    :return: all possible allocations (paths through the DAG)
    """
    return find_subpaths_allocations(adj_mat, 0, dest, {}, battlefields, N)


def compute_expected_payoff(decision, opp_decisions, win_draws=False, divide=True):
    """
    Compute the expected payoff for a given decision
    :param decision: decision to compute the expected payoff of
    :param opp_decisions: all possible decisions the opponent can take
    :param win_draws: whether the player wins draws
    :param divide: whether to divide the total available payoff by the number of possibilities (expected value)
    :return: the expected payoff (total available payoff if divide is False)
    """
    compare = np.greater_equal if win_draws else np.greater
    total = sum(compare(dec, decision).sum() for dec in opp_decisions)
    return total/len(opp_decisions) if divide else total



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

