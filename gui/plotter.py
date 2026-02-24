"""
gui/plotter.py
"""
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List


class ProbabilityPlotter:
    def __init__(self, parent_frame: tk.Frame):
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.figure.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._format_axes()
        self.canvas.draw()  # Draw once on init

    def _format_axes(self):
        """Prepares the axes. Does not trigger a canvas redraw to save UI resources."""
        self.ax.clear()
        self.ax.set_title("Probability of Rain Over Time", fontsize=13, fontweight='bold', pad=10)
        self.ax.set_xlabel("Timeline (Days)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Probability of Rain (%)", fontsize=10, fontweight='bold')
        self.ax.set_ylim(-5, 115)  # Leave room at the top for labels
        self.ax.grid(True, linestyle='--', alpha=0.4)

    def update_plot(self, days: List[int], forward_probs: List[float], smoothed_probs: List[float], lag: int):
        self._format_axes()
        if not days:
            self.canvas.draw()
            return

        # Convert raw probabilities to percentages
        f_pct = [p * 100 for p in forward_probs]
        s_pct = [p * 100 for p in smoothed_probs]

        # 1. Draw the Shaded "Active Memory" Window
        current_day = days[-1]
        window_start = max(1, current_day - lag)
        self.ax.axvspan(window_start, current_day, color='#f0f0f0', alpha=0.8, label=f'Memory Window (d={lag})')

        # 2. Plot Forward Filter
        self.ax.plot(days, f_pct, color='#1f77b4', linestyle='--', marker='o', alpha=0.6, label='Present Belief')

        # Annotate the newest forward point
        self.ax.annotate(f"{f_pct[-1]:.1f}%", (days[-1], f_pct[-1]),
                         textcoords="offset points", xytext=(0, 10), ha='center', color='#1f77b4', fontsize=9)

        # 3. Plot Smoothed Line (Robustly matched to the length of available data)
        if s_pct:
            # Safely slice the days array to match exactly how many smoothed points we have
            smoothed_days = days[:len(s_pct)]

            self.ax.plot(smoothed_days, s_pct, color='#d62728', linestyle='-', linewidth=2, marker='s',
                         label='Smoothed Past')

            # Annotate the newest smoothed point
            self.ax.annotate(f"{s_pct[-1]:.1f}%", (smoothed_days[-1], s_pct[-1]),
                             textcoords="offset points", xytext=(0, -15), ha='center', color='#d62728',
                             fontweight='bold', fontsize=9)

        # Force integer ticks on the X axis
        self.ax.set_xticks(range(max(1, min(days)), max(days) + 2))
        self.ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Draw everything to the screen once
        self.canvas.draw()
