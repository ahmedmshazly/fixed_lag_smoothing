"""
world/hmm.py

A container grouping the world's transition rules, sensor rules, and initial belief.
"""
import numpy as np
from typing import List
from world.transition import TransitionModel
from world.sensor import SensorModel


class HiddenMarkovModel:
    """
    Represents the complete Hidden Markov Model environment.
    """

    def __init__(self, transition_model: TransitionModel, sensor_model: SensorModel, prior: List[float]):
        """
        Initializes the HMM (I like to call it HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM) and validates that all components align mathematically.

        Args:
            transition_model: An instance of TransitionModel.
            sensor_model: An instance of SensorModel.
            prior: A list of initial probabilities for each hidden state (Day 0).
        """
        self.transition_model = transition_model
        self.sensor_model = sensor_model

        # 1. Validate alignment between transition and sensor models
        t_states = self.transition_model.get_T().shape[0]
        s_states = self.sensor_model.num_states
        if t_states != s_states:
            raise ValueError(f"Model mismatch! Transition has {t_states} states, but Sensor has {s_states} states.")

        self.num_states = t_states  # Aligned

        # 2. Validate the Prior distribution
        self.prior = np.array(prior, dtype=float)

        if len(self.prior) != self.num_states:
            raise ValueError(
                f"Prior length ({len(self.prior)}) must match the number of hidden states ({self.num_states}).")

        if not np.isclose(np.sum(self.prior), 1.0):
            raise ValueError(f"The prior distribution must sum to 1.0. Current sum: {np.sum(self.prior)}")

        if np.any(self.prior < 0.0):
            raise ValueError("Prior probabilities cannot be negative.")

    def get_prior(self) -> np.ndarray:
        """Returns the Day 0 initial belief vector."""
        return self.prior
