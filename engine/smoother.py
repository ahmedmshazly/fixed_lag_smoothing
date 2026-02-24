"""
engine/smoother.py

The orchestrator for the Fixed-Lag Smoothing algorithm.
"""
import numpy as np
from collections import deque
from typing import Any, Optional

from world.hmm import HiddenMarkovModel
from engine.window import EvidenceWindow
from engine.forward import ForwardFilter
from engine.backward import BackwardTransformer


class FixedLagSmoother:
    """
    Executes a stable, modular Fixed-Lag Smoothing algorithm.
    """

    def __init__(self, hmm: HiddenMarkovModel, lag: int):
        self.hmm = hmm
        self.lag = lag
        self.t = 1

        self.window = EvidenceWindow(lag=self.lag)
        self.forward_filter = ForwardFilter(prior=self.hmm.get_prior())

        self.T = self.hmm.transition_model.get_T()
        self.T_transposed = self.hmm.transition_model.get_T_transposed()

        # Cleanly instantiate our separated backward logic
        self.backward_transformer = BackwardTransformer(
            num_states=self.hmm.num_states,
            transition_matrix=self.T
        )

        # A small memory buffer to hold the last 'd' forward messages.
        self.f_history = deque([self.hmm.get_prior()], maxlen=self.lag + 1)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Helper to scale a vector so its probabilities sum to 1.0."""
        total = np.sum(vector)
        if total == 0:
            return vector
        return vector / total

    def _format_prob(self, vector: np.ndarray) -> str:
        """Helper to translate a [Rain, Sun] array into a readable string."""
        if vector is None:
            return "N/A"
        return f"{vector[0] * 100:.2f}% Rain, {vector[1] * 100:.2f}% Sun"

    def process_day(self, evidence: Any) -> Optional[np.ndarray]:
        """Processes a single day with highly educational, conceptual logging."""

        evidence_str = "Umbrella Seen" if evidence else "No Umbrella"

        print(f"\n{'=' * 60}")
        print(f" 📅 DAY {self.t} | Present Evidence: {evidence_str} ({evidence})")
        print(f"{'=' * 60}")

        # --- PHASE 1: THE PRESENT ---
        print("\n[1] PRESENT TIMELINE (Filtering)")
        self.window.add(evidence)
        O_t = self.hmm.sensor_model.get_O(evidence)

        self.forward_filter.step_forward(self.T_transposed, O_t)
        f_t = self.forward_filter.get_current_f()
        self.f_history.append(f_t)
        print("    -> Pushing yesterday's belief through weather and sensor rules.")
        print(f"    -> Today's Present Belief:  {self._format_prob(f_t)}")

        # --- PHASE 2: MEMORY STATE ---
        print("\n[2] MEMORY WINDOW")
        print(f"    -> Current Window Contents: {self.window.get_contents()}")

        # --- PHASE 3: STABLE SMOOTHING (The Backward Pass) ---
        result = None
        if self.t >= self.lag + 1:
            smoothed_day = self.t - self.lag
            print(f"\n[3] SMOOTHING THE PAST (Revisiting Day {smoothed_day})")

            # Delegate the heavy mathematical lifting to the BackwardTransformer
            b = self.backward_transformer.compute_backward_message(
                window_evidence=self.window.get_contents(),
                sensor_model=self.hmm.sensor_model
            )

            f_t_minus_d = self.f_history[0]

            # Combine the forward message from the past with the backward context from the future
            unnormalized_smoothed = f_t_minus_d * b

            if np.sum(unnormalized_smoothed) == 0:
                result = f_t_minus_d
            else:
                result = self._normalize(unnormalized_smoothed)

            print(f"    -> Base belief on Day {smoothed_day} was: {self._format_prob(f_t_minus_d)}")
            print(f"    -> Applying future context backwards through the window...")
            print(f"    -> 🔮 UPDATED PAST BELIEF:   {self._format_prob(result)}")

            # Slide the window forward to respect the lag limit
            self.window.pop_oldest()

        self.t += 1
        return result
