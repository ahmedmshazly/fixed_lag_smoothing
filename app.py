"""
app.py
The main entry point for the Fixed-Lag Smoothing Simulation application.
Wires up the Model-View-Controller (MVC) components and starts the GUI event loop.
"""
import tkinter as tk
from tkinter import ttk

from gui.controllers.simulation_controller import SimulationController
from gui.views.main_window import MainWindow


def main() -> None:
    """Initialize the application components and start the main event loop."""
    # 1. Initialize the Tkinter root window
    root = tk.Tk()

    # 2. Set a clean, modern theme if available on the host OS
    try:
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    except Exception as e:
        print(f"Notice: Could not apply 'clam' theme. Using default OS theme. ({e})")

    # 3. Instantiate the Controller (Business Logic & Engine Management)
    controller = SimulationController()

    # 4. Instantiate the View (GUI) and inject the Controller (Dependency Injection)
    app = MainWindow(root, controller)

    # 5. Start the application event loop
    root.mainloop()


if __name__ == "__main__":
    main()
