"""
world/transition.py

Handles the state-to-state transition rules.
"""
import numpy as np


class TransitionModel:
    """
    Represents the transition rules of a Hidden Markov Model.
    Stores the transition matrix <T> and its <transpose>.
    """

    def __init__(self, transition_matrix: list | np.ndarray):
        """
        Initializes the TransitionModel with validation.

        Args:
            transition_matrix: A 2D list or numpy array representing <T> transition matrix.
                               T[i][j] is the probability of moving from state i to state j.
        """
        self.T = np.array(transition_matrix, dtype=float)

        # 1. Validate Shape
        if self.T.ndim != 2 or self.T.shape[0] != self.T.shape[1]:
            raise ValueError(f"Transition matrix must be a square 2D array. Got shape: {self.T.shape}")

        # 2. Validate Probabilities (each row must sum to 1.0)
        row_sums = np.sum(self.T, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"Every row in the transition matrix must sum to 1.0. Row sums: {row_sums}")

        # Pre-compute the Transpose matrix (Cause it is used heavily in the Forward algorithm filtering step)
        self.T_transposed = self.T.T

    def get_T(self) -> np.ndarray:
        """Returns the standard transition matrix."""
        return self.T

    def get_T_transposed(self) -> np.ndarray:
        """Returns the transposed transition matrix."""
        return self.T_transposed
