"""
gui/__init__.py

Public API for the GUI domain.
"""
from gui.views.main_window import MainWindow
from .plotter import ProbabilityPlotter

__all__ = ["MainWindow", "ProbabilityPlotter"]
