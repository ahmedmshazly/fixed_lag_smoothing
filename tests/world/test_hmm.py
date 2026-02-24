"""
tests/world/test_hmm.py

Unit tests for the HiddenMarkovModel class.
"""
import unittest
import numpy as np
from world.transition import TransitionModel
from world.sensor import SensorModel
from world.hmm import HiddenMarkovModel


class TestHiddenMarkovModel(unittest.TestCase):

    def setUp(self):
        """Set up valid Umbrella World components before each test runs."""
        self.valid_transition = TransitionModel([[0.7, 0.3], [0.3, 0.7]])
        self.valid_sensor = SensorModel({True: [0.9, 0.2], False: [0.1, 0.8]})
        self.valid_prior = [0.5, 0.5]

    def test_valid_initialization(self):
        """Test that the HMM successfully groups valid components."""
        hmm = HiddenMarkovModel(self.valid_transition, self.valid_sensor, self.valid_prior)

        self.assertEqual(hmm.num_states, 2)
        np.testing.assert_array_almost_equal(hmm.get_prior(), np.array([0.5, 0.5]))
        self.assertIsInstance(hmm.transition_model, TransitionModel)
        self.assertIsInstance(hmm.sensor_model, SensorModel)

    def test_mismatched_states_raises_error(self):
        """Test that passing models with different state counts raises a ValueError."""
        # A 3-state transition model (e.g., Rain, Sun, Snow)
        bad_transition = TransitionModel([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])

        with self.assertRaisesRegex(ValueError, "Model mismatch"):
            HiddenMarkovModel(bad_transition, self.valid_sensor, [0.33, 0.33, 0.34])

    def test_invalid_prior_length_raises_error(self):
        """Test that a prior with the wrong number of elements raises a ValueError."""
        bad_prior = [0.5, 0.3, 0.2]  # 3 items, but our world only has 2 states

        with self.assertRaisesRegex(ValueError, "Prior length"):
            HiddenMarkovModel(self.valid_transition, self.valid_sensor, bad_prior)

    def test_invalid_prior_sum_raises_error(self):
        """Test that a prior that does not sum to 1.0 raises a ValueError."""
        bad_prior = [0.8, 0.8]  # Sums to 1.6

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            HiddenMarkovModel(self.valid_transition, self.valid_sensor, bad_prior)


if __name__ == '__main__':
    unittest.main()