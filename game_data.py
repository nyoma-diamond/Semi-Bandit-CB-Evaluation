from dataclasses import dataclass

import numpy as np

@dataclass
class GameData:
    K: int
    T: int

    A_algorithm: str
    A_resources: int
    A_decisions: np.ndarray
    A_results: np.ndarray

    B_algorithm: str
    B_resources: int
    B_decisions: np.ndarray
    B_results: np.ndarray

    def identifier(self) -> str:
        """
        Creates a filename-safe identifier
        :return: filename-safe identifier
        """
        return '-'.join(str(x) for x in [self.T, self.K, self.A_algorithm, self.A_resources, self.B_algorithm, self.B_resources])

    def is_valid(self) -> bool:
        """
        returns whether the game is valid or not
        :return: game validity
        """
        return len(self.A_decisions) == len(self.B_decisions) == self.T

def parse_identifier(identifier: str) -> GameData:
    """
    Parses the provided filename identifier
    Note: this is effectively the inverse of the identifier() function in GameData
    :param identifier: identifier to parse
    :return: a corresponding GameData instance
    """
    split = identifier.split('-')
    return GameData(int(split[1]), int(split[0]),
                    split[2], int(split[3]),
                    np.empty(0), np.empty(0),
                    split[4], int(split[5]),
                    np.empty(0), np.empty(0))