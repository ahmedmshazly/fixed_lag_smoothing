"""
tests/world/test_transition.py

Unit tests for the TransitionModel class.
"""
import unittest
import numpy as np
from world.transition import TransitionModel


class TestTransitionModel(unittest.TestCase):

    def setUp(self):
        """Set up valid data before each test runs."""
        # The Umbrella World transition matrix
        self.valid_matrix = [
            [0.7, 0.3],
            [0.3, 0.7]
        ]

    def test_valid_initialization(self):
        """Test that a correct matrix initializes without errors and stores correctly."""
        model = TransitionModel(self.valid_matrix)

        # Check that T is stored as a numpy array
        self.assertIsInstance(model.get_T(), np.ndarray)
        # Check shape
        self.assertEqual(model.get_T().shape, (2, 2))

    def test_transpose_calculation(self):
        """Test that the transpose is calculated correctly."""
        model = TransitionModel(self.valid_matrix)
        expected_transpose = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        # np.testing.assert_array_almost_equal is best for comparing float matrices
        np.testing.assert_array_almost_equal(model.get_T_transposed(), expected_transpose)

    def test_inverse_calculation(self):
        """Test that the inverse is calculated correctly."""
        model = TransitionModel(self.valid_matrix)
        # The mathematical inverse of [[0.7, 0.3], [0.3, 0.7]] is:
        # [[ 1.75, -0.75], [-0.75,  1.75]]
        expected_inverse = np.array([
            [1.75, -0.75],
            [-0.75, 1.75]
        ])
        np.testing.assert_array_almost_equal(model.get_T_inverse(), expected_inverse)

    def test_invalid_shape_raises_error(self):
        """Test that a non-square matrix raises a ValueError."""
        bad_matrix = [
            [0.7, 0.3, 0.1],
            [0.3, 0.7, 0.2]
        ]
        with self.assertRaisesRegex(ValueError, "must be a square 2D array"):
            TransitionModel(bad_matrix)

    def test_invalid_probabilities_raises_error(self):
        """Test that rows not summing to 1.0 raise a ValueError."""
        bad_matrix = [
            [0.8, 0.5],  # Sums to 1.3
            [0.3, 0.7]
        ]
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            TransitionModel(bad_matrix)

    def test_singular_matrix_raises_error(self):
        """Test that a matrix which cannot be inverted raises a ValueError."""
        # A matrix where rows are identical has a determinant of 0 and cannot be inverted.
        singular_matrix = [
            [0.5, 0.5],
            [0.5, 0.5]
        ]
        with self.assertRaisesRegex(ValueError, "singular and cannot be inverted"):
            TransitionModel(singular_matrix)


if __name__ == '__main__':
    unittest.main()