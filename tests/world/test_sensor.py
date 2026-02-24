"""
tests/world/test_sensor.py

Unit tests for the SensorModel class.
"""
import unittest
import numpy as np
from world.sensor import SensorModel


class TestSensorModel(unittest.TestCase):

    def setUp(self):
        """Set up valid Umbrella World data before each test runs."""
        # True = Umbrella seen, False = No umbrella seen
        # Probabilities correspond to [Rain, Sun]
        self.valid_probs = {
            True: [0.9, 0.2],
            False: [0.1, 0.8]
        }

    def test_valid_initialization(self):
        """Test that correct probabilities initialize without errors."""
        model = SensorModel(self.valid_probs)
        self.assertEqual(model.num_states, 2)

    def test_get_O_matrix(self):
        """Test that the diagonal matrix O_t is built correctly."""
        model = SensorModel(self.valid_probs)

        expected_O_true = np.array([
            [0.9, 0.0],
            [0.0, 0.2]
        ])
        np.testing.assert_array_almost_equal(model.get_O(True), expected_O_true)

    def test_get_O_inverse_matrix(self):
        """Test that the inverted diagonal matrix O_t^(-1) is built correctly."""
        model = SensorModel(self.valid_probs)

        expected_O_inv_true = np.array([
            [1.0 / 0.9, 0.0],
            [0.0, 1.0 / 0.2]
        ])
        np.testing.assert_array_almost_equal(model.get_O_inverse(True), expected_O_inv_true)

    def test_unrecognized_evidence_raises_error(self):
        """Test that asking for evidence not in the dictionary raises a KeyError."""
        model = SensorModel(self.valid_probs)
        with self.assertRaisesRegex(KeyError, "not recognized"):
            model.get_O("Raincoat")

    def test_zero_probability_raises_error(self):
        """Test that a probability of 0.0 raises a ValueError to protect invertibility."""
        bad_probs = {
            True: [1.0, 0.0],  # 0.0 is illegal for our smoother
            False: [0.0, 1.0]
        }
        with self.assertRaisesRegex(ValueError, "strictly greater than 0.0"):
            SensorModel(bad_probs)

    def test_inconsistent_states_raises_error(self):
        """Test that mismatched state lengths raise a ValueError."""
        bad_probs = {
            True: [0.9, 0.2],  # 2 states
            False: [0.1, 0.8, 0.5]  # 3 states (Invalid!)
        }
        with self.assertRaisesRegex(ValueError, "Inconsistent number of hidden states"):
            SensorModel(bad_probs)


if __name__ == '__main__':
    unittest.main()