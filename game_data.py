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

    def identifier(self):
        """
        Creates a filename-safe identifier
        :return: filename-safe identifier
        """
        return '-'.join(str(x) for x in [self.T, self.K, self.A_algorithm, self.A_resources, self.B_algorithm, self.B_resources])

    def is_valid(self):
        """
        returns whether the game is valid or not
        :return: game validity
        """
        return len(self.A_decisions) == len(self.B_decisions) == self.T