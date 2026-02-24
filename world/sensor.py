"""
world/sensor.py

Handles how hidden states produce visible evidence.
"""
import numpy as np
from typing import Dict, Any, List


class SensorModel:
    """
    Represents the observation (known as sensor) rules of a HMM.
    Generates the diagonal matrix <O_t> based on observed evidence.
    """

    def __init__(self, observation_probs: Dict[Any, List[float]]):
        """
        Initializes the SensorModel with validation.

        Args:
            observation_probs: A dictionary mapping an evidence state (like True/False)
                               to a list of probabilities for each hidden state.
                               Example: {True/Umbrella Seen: [0.9, 0.2], False/No Umbrella Seen: [0.1, 0.8]}
        """
        self.probs = {}

        # 1. Validate State Consistency
        # Ensure all probability lists have the same length (the number of hidden states, S)
        lengths = {len(v) for v in observation_probs.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"Inconsistent number of hidden states in observation probabilities. Lengths found: {lengths}")

        self.num_states = lengths.pop() if lengths else 0

        # 2. Validate bounds (0.0 to 1.0)
        # UPDATE: no longer need to ban 0.0 probabilities because our new mathematically stable smoothing engine does
        # not rely on inverted sensor matrices! Check the smoothing engine to understand.
        for evidence, prob_list in observation_probs.items():
            prob_array = np.array(prob_list, dtype=float)

            if np.any((prob_array < 0.0) | (prob_array > 1.0)):
                raise ValueError(
                    f"Probabilities must be between 0.0 and 1.0 inclusive. "
                    f"Found {prob_array} for evidence '{evidence}'."
                )

            self.probs[evidence] = prob_array

    def get_O(self, evidence: Any) -> np.ndarray:
        """
        Builds the diagonal sensor matrix O_t for the given evidence.
        """
        if evidence not in self.probs:
            raise KeyError(f"Evidence '{evidence}' is not recognized by the SensorModel.")

        # np.diag takes a 1D array and puts it on the diagonal of a square matrix
        return np.diag(self.probs[evidence])
