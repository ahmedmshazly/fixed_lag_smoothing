"""
engine/window.py

Manages the sliding t-d to t timeline of evidence.
"""
from collections import deque
from typing import Any


class EvidenceWindow:
    """
    A specialized queue that strictly manages the sliding window of evidence.
    It holds exactly (d + 1) days of evidence before it is considered 'full'.
    """

    def __init__(self, lag: int):
        """
        Initializes the EvidenceWindow.

        Args:
            lag (int): The number of days in the past we want to smooth (d).
        """
        if lag < 1:
            raise ValueError(f"Smoothing lag 'd' must be at least 1. Got: {lag}")

        self.lag = lag
        # The window needs to hold evidence from t-d all the way to t inclusive.
        # This means it holds (lag + 1) items.
        self.max_size = lag + 1

        # A deque is much faster than a list.
        self.queue = deque()

    def add(self, evidence: Any) -> None:
        """Adds a new day's evidence to the front (right side) of the queue."""
        self.queue.append(evidence)

    def is_full(self) -> bool:
        """
        Checks if the window has collected enough evidence to perform smoothing.
        This represents the 'if t > d' logic in the book.
        """
        return len(self.queue) == self.max_size

    def pop_oldest(self) -> Any:
        """
        Removes and returns the oldest piece of evidence from the back (left side).
        This executes the 'remove e_{t-d-1}' logic from the book.
        """
        if not self.queue:
            raise IndexError("Cannot pop from an empty EvidenceWindow.")
        return self.queue.popleft()

    def get_oldest(self) -> Any:
        """
        Returns the oldest piece of evidence without removing it.
        We need this to build O_{t-d} before we slide the window.
        """
        if not self.queue:
            raise IndexError("EvidenceWindow is empty.")
        return self.queue[0]

    def __len__(self) -> int:
        """Returns the current number of items in the window."""
        return len(self.queue)

    def get_contents(self) -> list:
        """Returns the current timeline of evidence as a standard list."""
        return list(self.queue)
