"""
engine/backward.py

Executes the mathematically stable backward message calculations, essential part for the smoother.
"""
import numpy as np
from typing import List, Any


class BackwardTransformer:
    """
    Computes the backward message 'b' over a fixed memory window (number of days in our world case).
    Uses the stable standard backward algorithm to completely prevent numerical collapse and catastrophic cancellation.
    """

    def __init__(self, num_states: int, transition_matrix: np.ndarray):
        """
        Initializes the backward transformer.

        Args:
            num_states: The number of hidden states (S) in our world (e.g., Sunny/Rainy).
            transition_matrix: The standard transition matrix (T).
        """
        if num_states < 1:
            raise ValueError("Number of states must be at least 1.")

        self.num_states = num_states
        self.T = transition_matrix

    def compute_backward_message(self, window_evidence: List[Any], sensor_model: Any) -> np.ndarray:
        """
        Calculates the backward message vector for the oldest day in the window.
        Steps backward from the present day (t) down to the smoothed day (t-d).
        """
        print("    -> [DEBUG] Packing the Time Traveler's Envelope (Backward Message 'b')")

        # Start the backward message at the present day with [1.0, 1.0, ...]
        b = np.ones(self.num_states)
        print(f"    -> Starting 'b' (Present Day): [{b[0]:.4f}, {b[1]:.4f}]")

        # Step backward through the exact evidence in our window
        # Reversing the list to go back in time from day t down to day t-d+1
        for i, evidence in enumerate(reversed(window_evidence)):
            O_k = sensor_model.get_O(evidence)

            print(f"      --- Time Travel Step {i + 1} (Evidence: {evidence}) ---")
            print(f"      Matrix O_k applied: diag({O_k[0, 0]:.2f}, {O_k[1, 1]:.2f})")

            # The stable backward formula
            b_raw = self.T @ O_k @ b
            print(f"      Raw 'b' after pushing through rules: [{b_raw[0]:.4f}, {b_raw[1]:.4f}]")

            # Normalize 'b' at each step to strictly prevent floating-point explosion or collapse
            total = np.sum(b_raw)
            if total > 0:
                b = b_raw / total
            else:
                b = b_raw

            print(f"      Normalized 'b' (Ready for next step): [{b[0]:.4f}, {b[1]:.4f}]")

        print("    -> [DEBUG] Envelope packed! Ready to multiply with Past Belief.")
        return b
