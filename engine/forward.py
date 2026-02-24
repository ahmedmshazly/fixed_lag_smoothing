"""
engine/forward.py

Executes the forward message filtering math.
"""
import numpy as np


class ForwardFilter:
    """
    Manages and updates the forward message 'f' over time.
    Calculates the present-day belief using the Forward algorithm.
    """

    def __init__(self, prior: np.ndarray):
        """
        Initializes the forward filter with the Day 0 prior belief.

        Args:
            prior: A 1D numpy array representing the initial belief distribution.
        """
        # Copy the array to prevent accidental modification of the original prior
        self.f = np.copy(prior)

        if not np.isclose(np.sum(self.f), 1.0):
            raise ValueError("The initial forward message (prior) must sum to 1.0.")

    def step_forward(self, T_transposed: np.ndarray, O_t: np.ndarray) -> np.ndarray:
        """
        Pushes the belief forward by one day.
        Executes: f_{t+1} = alpha * O_{t+1} * T^T * f_t

        Args:
            T_transposed: The transposed transition matrix.
            O_t: The diagonal sensor matrix for today's evidence.

        Returns:
            The newly updated forward message vector.
        """
        # Step 1 & 2: Push yesterday's belief through the weather rules
        predicted_belief = np.dot(T_transposed, self.f)

        # Step 3: Filter the prediction through today's sensor evidence
        unnormalized_f = np.dot(O_t, predicted_belief)

        # Step 4: Normalize (the 'alpha' constant)
        total_prob = np.sum(unnormalized_f)

        if total_prob == 0:
            raise ValueError(
                "Forward message probabilities summed to 0. "
                "This means the model believes the current sequence of events is strictly impossible."
            )

        self.f = unnormalized_f / total_prob  # Normalized by division.

        return self.f

    def get_current_f(self) -> np.ndarray:
        """Returns the current state of the forward message."""
        return self.f
