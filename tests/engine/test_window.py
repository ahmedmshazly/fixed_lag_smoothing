"""
tests/engine/test_window.py

Unit tests for the EvidenceWindow class.
"""
import unittest
from engine.window import EvidenceWindow


class TestEvidenceWindow(unittest.TestCase):

    def test_valid_initialization(self):
        """Test that the window calculates its max_size correctly."""
        # If lag d=2, max_size should be 3 (t-2, t-1, t)
        window = EvidenceWindow(lag=2)
        self.assertEqual(window.lag, 2)
        self.assertEqual(window.max_size, 3)
        self.assertEqual(len(window), 0)

    def test_invalid_lag_raises_error(self):
        """Test that a lag of 0 or less raises a ValueError."""
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            EvidenceWindow(lag=0)

    def test_window_filling_and_is_full(self):
        """Test the lifecycle of adding items and checking if full."""
        window = EvidenceWindow(lag=2)  # Needs 3 items to be full

        window.add("Day 1")
        self.assertFalse(window.is_full())

        window.add("Day 2")
        self.assertFalse(window.is_full())

        window.add("Day 3")
        self.assertTrue(window.is_full())  # Now it has 3 items!

    def test_pop_oldest_removes_and_returns(self):
        """Test that popping from the left works exactly as expected."""
        window = EvidenceWindow(lag=2)
        window.add("Day 1")
        window.add("Day 2")
        window.add("Day 3")

        # Pop should return "Day 1" and reduce the length
        oldest = window.pop_oldest()
        self.assertEqual(oldest, "Day 1")
        self.assertEqual(len(window), 2)
        self.assertFalse(window.is_full())  # No longer full

    def test_get_oldest_does_not_remove(self):
        """Test that get_oldest peeks at the front without shrinking the queue."""
        window = EvidenceWindow(lag=2)
        window.add("Day 1")

        oldest = window.get_oldest()
        self.assertEqual(oldest, "Day 1")
        self.assertEqual(len(window), 1)  # Length should remain unchanged

    def test_pop_empty_raises_error(self):
        """Test that trying to pop or peek an empty window raises an IndexError."""
        window = EvidenceWindow(lag=2)
        with self.assertRaisesRegex(IndexError, "empty"):
            window.pop_oldest()

        with self.assertRaisesRegex(IndexError, "empty"):
            window.get_oldest()


if __name__ == '__main__':
    unittest.main()