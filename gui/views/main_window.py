"""
gui/views/main_window.py
Strictly handles UI layout, component mapping, and user input delegation.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from gui.styles import UITheme
from gui.utils.markdown_renderer import MarkdownRenderer
from gui.controllers.simulation_controller import SimulationController
from gui.plotter import ProbabilityPlotter
from gui.views.math_view import MathDetailsWindow


class MainWindow:
    """Expert-grade View component."""

    def __init__(self, root: tk.Tk, controller: SimulationController) -> None:
        self.root = root
        self.controller = controller

        self.root.title("Fixed-Lag Smoothing Simulation")
        self.root.geometry("1100x750")

        self._init_vars()
        self._bind_controller()
        self._build_ui()

        # Trigger initial state
        self._on_apply_config()

    def _init_vars(self) -> None:
        self.lag_var = tk.IntVar(value=2)
        self.sensor_var = tk.DoubleVar(value=0.90)
        self.transition_var = tk.DoubleVar(value=0.70)
        self.false_alarm_var = tk.DoubleVar(value=0.20)  # NEW: False alarm variable
        self.insight_text = tk.StringVar(value="Initializing...")

    def _bind_controller(self) -> None:
        """Connects the controller's outbound signals to UI update methods."""
        self.controller.on_log_update = self.append_log
        self.controller.on_insight_update = self.insight_text.set
        self.controller.on_error = self.show_error
        # Wait until UI is built to bind plot updates

    def _build_ui(self) -> None:
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_config_panel(left_frame)
        self._build_evidence_panel(left_frame)
        self._build_insight_panel(left_frame)
        self._build_log_panel(left_frame)

        self.plotter = ProbabilityPlotter(right_frame)
        self.controller.on_plot_update = self.plotter.update_plot

    def _build_config_panel(self, parent: tk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="⚙️ World Configuration", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="Lag Window (d):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.lag_var, width=5).grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Sensor Accuracy:").grid(row=1, column=0, sticky="w")
        tk.Scale(frame, from_=0.5, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.sensor_var, showvalue=True).grid(row=1, column=1)

        ttk.Label(frame, text="Weather Persistence:").grid(row=2, column=0, sticky="w")
        tk.Scale(frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.transition_var, showvalue=True).grid(row=2, column=1)

        # NEW: False Alarm Slider
        ttk.Label(frame, text="False Alarm Rate:").grid(row=3, column=0, sticky="w")
        tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.false_alarm_var, showvalue=True).grid(row=3, column=1)

        # Shifted the Apply button down to row 4
        ttk.Button(frame, text="Apply & Reset World", command=self._on_apply_config).grid(row=4, column=0, columnspan=2,
                                                                                          pady=10)

    def _build_evidence_panel(self, parent: tk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="📅 Add Daily Evidence", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Note: Depending on how your Controller handles evidence, it might be cleaner to just pass the boolean
        # and let the Engine use the configured values, but we will leave this passing sensor_var as is for now.
        ttk.Button(frame, text="☂️ Saw Umbrella",
                   command=lambda: self.controller.process_evidence(True, self.sensor_var.get())).pack(side=tk.LEFT,
                                                                                                       expand=True,
                                                                                                       fill=tk.X,
                                                                                                       padx=2)
        ttk.Button(frame, text="☀️ No Umbrella",
                   command=lambda: self.controller.process_evidence(False, self.sensor_var.get())).pack(side=tk.LEFT,
                                                                                                        expand=True,
                                                                                                        fill=tk.X,
                                                                                                        padx=2)

    def _build_insight_panel(self, parent: tk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="💡 AI Insight", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, textvariable=self.insight_text, wraplength=350,
                  foreground=UITheme.COLORS["insight_fg"], font=UITheme.FONTS["italic"]).pack(fill=tk.X)

        ttk.Button(frame, text="📐 View Mathematical Details",
                   command=self._show_math_details).pack(pady=(5, 0))

    def _build_log_panel(self, parent: tk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Mathematical Terminal Log", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.text_log = tk.Text(frame, wrap=tk.WORD, font=UITheme.FONTS["mono"], state=tk.DISABLED,
                                bg=UITheme.COLORS["bg_light"])
        scrollbar = ttk.Scrollbar(frame, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=scrollbar.set)

        UITheme.apply_markdown_tags(self.text_log)

        self.text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # --- UI Action Delegates ---

    def _on_apply_config(self) -> None:
        self.text_log.configure(state=tk.NORMAL)
        self.text_log.delete(1.0, tk.END)
        self.text_log.configure(state=tk.DISABLED)

        # NEW: Now passing 4 parameters to the controller instead of 3
        self.controller.initialize_world(
            self.lag_var.get(),
            self.sensor_var.get(),
            self.transition_var.get(),
            self.false_alarm_var.get()
        )

    def _show_math_details(self) -> None:
        """Opens the mathematical details window."""
        MathDetailsWindow(self.root)

    def append_log(self, text: str) -> None:
        MarkdownRenderer.render(text, self.text_log)

    def show_error(self, context: str, exception: Exception) -> None:
        error_msg = f"---\n⚠️ **{context}**\n*{exception.__class__.__name__}: {exception}*\n"
        self.append_log(error_msg)
        self.insight_text.set(f"⚠️ Error: Check log for details.")
        messagebox.showerror("Simulation Error", f"{context}\n{exception}")


# To run:
if __name__ == "__main__":
    root = tk.Tk()
    ctrl = SimulationController()
    app = MainWindow(root, ctrl)
    root.mainloop()
