"""
tests/engine/test_forward.py

Unit tests for the ForwardFilter class.
"""
import unittest
import numpy as np
from engine.forward import ForwardFilter


class TestForwardFilter(unittest.TestCase):

    def setUp(self):
        """Set up standard matrices for the Umbrella World."""
        self.prior = np.array([0.5, 0.5])

        # Transition matrix: [[0.7, 0.3], [0.3, 0.7]]
        # The transpose is exactly the same because it's symmetrical!
        self.T_transposed = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])

        # Sensor matrix for Umbrella = True
        self.O_true = np.array([
            [0.9, 0.0],
            [0.0, 0.2]
        ])

    def test_valid_initialization(self):
        """Test that it stores the prior correctly."""
        filter_engine = ForwardFilter(self.prior)
        np.testing.assert_array_almost_equal(filter_engine.get_current_f(), self.prior)

    def test_invalid_prior_raises_error(self):
        """Test that a non-normalized prior raises an error."""
        bad_prior = np.array([0.8, 0.8])
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            ForwardFilter(bad_prior)

    def test_step_forward_math(self):
        """
        Test the mathematical exactness of the forward step.
        Math walkthrough for Day 1:
        1. T^T * f = [[0.7, 0.3], [0.3, 0.7]] * [0.5, 0.5] = [0.5, 0.5]
        2. O_t * [0.5, 0.5] = [[0.9, 0.0], [0.0, 0.2]] * [0.5, 0.5] = [0.45, 0.1]
        3. Normalize [0.45, 0.1]: sum is 0.55.
           0.45 / 0.55 = 0.8181818...
           0.1 / 0.55 = 0.1818181...
        """
        filter_engine = ForwardFilter(self.prior)

        expected_f = np.array([0.81818182, 0.18181818])

        new_f = filter_engine.step_forward(self.T_transposed, self.O_true)

        # Check that the returned value is correct
        np.testing.assert_array_almost_equal(new_f, expected_f)
        # Check that the internal state was updated correctly
        np.testing.assert_array_almost_equal(filter_engine.get_current_f(), expected_f)

    def test_zero_probability_raises_error(self):
        """Test that an impossible sequence raises a division-by-zero safeguard."""
        filter_engine = ForwardFilter(self.prior)

        # An impossible observation matrix (all zeros)
        O_impossible = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ])

        with self.assertRaisesRegex(ValueError, "summed to 0"):
            filter_engine.step_forward(self.T_transposed, O_impossible)


if __name__ == '__main__':
    unittest.main()