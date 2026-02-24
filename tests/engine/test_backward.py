"""
tests/engine/test_backward.py

Unit tests for the BackwardTransformer class.
"""
import unittest
import numpy as np
from engine.backward import BackwardTransformer

class TestBackwardTransformer(unittest.TestCase):

    def setUp(self):
        """Set up standard matrices for the Umbrella World."""
        # Standard rules
        self.T = np.array([[0.7, 0.3], [0.3, 0.7]])
        self.T_inv = np.array([[1.75, -0.75], [-0.75, 1.75]])

        # Sensor matrices
        self.O_true = np.array([[0.9, 0.0], [0.0, 0.2]])
        self.O_false = np.array([[0.1, 0.0], [0.0, 0.8]])

        # Inverted sensor matrices
        self.O_true_inv = np.array([[1.0/0.9, 0.0], [0.0, 1.0/0.2]])
        self.O_false_inv = np.array([[1.0/0.1, 0.0], [0.0, 1.0/0.8]])

    def test_valid_initialization(self):
        """Test that B initializes as an Identity Matrix."""
        transformer = BackwardTransformer(num_states=2)
        expected_B = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(transformer.get_B(), expected_B)

    def test_invalid_states_raises_error(self):
        with self.assertRaises(ValueError):
            BackwardTransformer(num_states=0)

    def test_grow_window(self):
        """Test B <- B * T * O_t"""
        transformer = BackwardTransformer(num_states=2)

        # Day 1: True
        # B = I * T * O_true
        expected_B = np.eye(2) @ self.T @ self.O_true
        new_B = transformer.grow_window(self.T, self.O_true)

        np.testing.assert_array_almost_equal(new_B, expected_B)
        np.testing.assert_array_almost_equal(transformer.get_B(), expected_B)

    def test_slide_window_math(self):
        """Test the constant-space un-multiplying hack."""
        transformer = BackwardTransformer(num_states=2)

        # Phase 1: Grow the window for Day 1 (True) and Day 2 (True)
        transformer.grow_window(self.T, self.O_true)
        B_after_day_2 = transformer.grow_window(self.T, self.O_true)

        # Phase 2: Slide the window for Day 3 (False).
        # This means we add Day 3 (False) to the front, and strip Day 1 (True) from the back.
        new_B = transformer.slide_window(
            T=self.T,
            T_inv=self.T_inv,
            O_t=self.O_false,
            O_t_minus_d_inv=self.O_true_inv
        )

        # Mathematically verify:
        # If the hack works, the new B should be EXACTLY equal to simply growing
        # a fresh window with only Day 2 and Day 3. Let's prove it!
        manual_transformer = BackwardTransformer(num_states=2)
        manual_transformer.grow_window(self.T, self.O_true)   # Add Day 2
        manual_expected_B = manual_transformer.grow_window(self.T, self.O_false) # Add Day 3

        np.testing.assert_array_almost_equal(new_B, manual_expected_B)

if __name__ == '__main__':
    unittest.main()