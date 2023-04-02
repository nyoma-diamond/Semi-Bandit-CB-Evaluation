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
