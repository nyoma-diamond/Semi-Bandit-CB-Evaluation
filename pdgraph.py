import numpy as np
import pandas as pd

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


def build_adjacency_matrix(battlefields: int, N: int, bounds: list[tuple[int,int]] = None) -> pd.DataFrame:
    """
    Creates DataFrame representing the adjacency matrix for possible allocations/decisions for a player
    :param N: the number of resources available to the player
    :param bounds: list of lower and upper bounds for possible allocations
    :return: Adjacency matrix (DataFrame) for graph representing possible allocations/decisions of style mat[from][to]
    """
    adj_mat = pd.DataFrame(-1,
                           index=pd.MultiIndex.from_tuples([index_to_coordinate(i, battlefields, N) for i in range(1, (battlefields-1) * (N+1) + 2)], names=['battlefield', 'resources used']),
                           columns=pd.MultiIndex.from_tuples([index_to_coordinate(i, battlefields, N) for i in range((battlefields-1) * (N+1) + 1)], names=['battlefield', 'resources used']))

    if bounds is None:
        bounds = [(0,N)]*battlefields

    for frm in adj_mat.columns:
        for to in adj_mat.loc[frm[0]+1].index:
            if to < frm[1]:
                continue
            allocation = to - frm[1]
            if bounds[frm[0]][0] <= allocation <= bounds[frm[0]][1]:
                adj_mat.at[(frm[0]+1, to), frm] = allocation

    return adj_mat


def find_subpaths(adj_mat: pd.DataFrame, node: tuple[int, int], dest: tuple[int, int],
                  visited: dict[tuple[int, int], list[list[tuple, tuple]]]):
    """
    Find all paths (subpaths) between the provided nodes in the provided DAG
    :param node: the starting node
    :param dest: the destination node
    :param visited: dictionary of all subpaths from nodes already visited
    :param adj_mat: adjacency matrix representing the DAG being searched
    :return: all paths (subpaths) between the provided nodes
    """
    if node not in visited.keys():
        visited[node] = [[node] + subpath
                         for child in adj_mat[adj_mat[node] != -1].index
                         for subpath in find_subpaths(adj_mat, child, dest, visited)]

    return visited[node]


def find_paths(adj_mat: pd.DataFrame, dest: tuple[int, int]):
    """
    Find all paths through the provided DAG
    :param dest: the destination node
    :param adj_mat: adjacency matrix representing the DAG being searched
    :return: all paths through the DAG
    """
    return find_subpaths(adj_mat, (0, 0), dest, {dest: [[dest]]})


def find_subpaths_allocations(adj_mat: pd.DataFrame, node: tuple[int, int], dest: tuple[int, int],
                              visited: dict[tuple[int, int], list[list[int]]]):
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
    if node not in visited.keys():
        visited[node] = [[adj_mat[node][child]] + subpath
                         for child in adj_mat[adj_mat[node] != -1].index
                         for subpath in find_subpaths_allocations(adj_mat, child, dest, visited)]

    return visited[node]


def find_paths_allocations(adj_mat: pd.DataFrame, dest: tuple[int, int]):
    """
    Find all possible paths (decisions) through the provided DAG
    :param dest: the destination node
    :param adj_mat: adjacency matrix representing the DAG being searched
    :return: all possible allocations (paths through the DAG)
    """
    return find_subpaths_allocations(adj_mat, (0, 0), dest, {})


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