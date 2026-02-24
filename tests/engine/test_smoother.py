"""
tests/engine/test_smoother.py

Unit tests for the FixedLagSmoother class.
"""
import unittest
import numpy as np
from world.transition import TransitionModel
from world.sensor import SensorModel
from world.hmm import HiddenMarkovModel
from engine.smoother import FixedLagSmoother


class TestFixedLagSmoother(unittest.TestCase):

    def setUp(self):
        """Set up a fully working Umbrella World."""
        transition = TransitionModel([[0.7, 0.3], [0.3, 0.7]])
        sensor = SensorModel({True: [0.9, 0.2], False: [0.1, 0.8]})
        prior = [0.5, 0.5]
        self.hmm = HiddenMarkovModel(transition, sensor, prior)

    def test_smoother_initialization(self):
        """Test that all components are wired up correctly."""
        smoother = FixedLagSmoother(self.hmm, lag=2)
        self.assertEqual(smoother.lag, 2)
        self.assertEqual(smoother.t, 1)
        self.assertEqual(len(smoother.f_history), 1)

    def test_process_day_returns_none_when_filling(self):
        """Test that process_day returns None when t <= d."""
        smoother = FixedLagSmoother(self.hmm, lag=2)

        # Day 1
        result = smoother.process_day(True)
        self.assertIsNone(result)
        self.assertEqual(smoother.t, 2)

    def test_process_day_returns_array_when_full(self):
        """Test that process_day returns a smoothed array when t > d."""
        smoother = FixedLagSmoother(self.hmm, lag=1)  # Lag is 1 for a quick test

        # Day 1: Filling
        smoother.process_day(True)

        # Day 2: Window is full, should return smoothed result for Day 1
        result = smoother.process_day(True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isclose(np.sum(result), 1.0))


if __name__ == '__main__':
    unittest.main()