"""
gui/views/math_view.py
A standalone view component for displaying mathematical equations and definitions.
"""
import tkinter as tk
from tkinter import ttk
from gui.styles import UITheme
from gui.utils.markdown_renderer import MarkdownRenderer


class MathDetailsWindow:
    """Creates a pop-up window containing the simulation's mathematical formulas."""

    def __init__(self, parent: tk.Tk) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("Mathematical Details – Fixed-Lag Smoothing")
        self.window.geometry("700x600")

        self._build_ui()

    def _build_ui(self) -> None:
        """Constructs the text widget and scrollbar for the equations."""
        text_frame = ttk.Frame(self.window, padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.text_widget = tk.Text(text_frame, wrap=tk.WORD, font=UITheme.FONTS["mono"],
                                   bg=UITheme.COLORS["bg_light"])
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        # Apply our centralized Markdown styles
        UITheme.apply_markdown_tags(self.text_widget)

        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Insert the equations and parse them
        equations = self._get_equations_text()
        MarkdownRenderer.render(equations, self.text_widget)
        self.text_widget.configure(state=tk.DISABLED)

        # Close button
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(pady=5)

    def _get_equations_text(self) -> str:
        """Returns the raw Markdown string containing all math definitions."""
        return r"""
## Hidden Markov Model (HMM) Equations

We model two hidden states: **Rain** (R) and **Sun** (S).  
Observations: **Umbrella** (U) or **No umbrella** (N).

### 1. Transition Model
*Probability of moving from one state to the next.*

$$
\mathbf{T} = \begin{bmatrix}
P(R_t|R_{t-1}) & P(S_t|R_{t-1}) \\
P(R_t|S_{t-1}) & P(S_t|S_{t-1})
\end{bmatrix}
= \begin{bmatrix}
a & 1-a \\
1-a & a
\end{bmatrix}
$$

- $a = P(R_t|R_{t-1}) = P(S_t|S_{t-1})$  (weather persistence, set by slider)
- $1-a$ = probability of switching state

### 2. Sensor (Observation) Model
*Probability of seeing evidence given the current state.*

$$
\mathbf{O} = \begin{bmatrix}
P(U|R) & P(N|R) \\
P(U|S) & P(N|S)
\end{bmatrix}
= \begin{bmatrix}
b & 1-b \\
0.2 & 0.8
\end{bmatrix}
$$

- $b = P(U|R)$ = sensor accuracy (set by slider)
- $P(U|S) = 0.2$ (fixed false positive rate)
- $P(N|S) = 0.8$ (true negative)
- $P(N|R) = 1-b$

### 3. Filtering (Forward Algorithm)
*Compute belief about current state given all evidence so far.*

Let $\mathbf{f}_{t} = P(X_t | e_{1:t})$ (vector of probabilities for Rain and Sun).

**Prediction step:**
$$
\overline{\mathbf{f}}_{t} = \mathbf{T}^\top \mathbf{f}_{t-1}
$$

**Update step (Bayes rule):**
$$
\mathbf{f}_{t} \propto \mathbf{O}_{e_t} \cdot \overline{\mathbf{f}}_{t}
$$

where $\mathbf{O}_{e_t}$ is the diagonal matrix of likelihoods for the observed evidence $e_t$.

### 4. Fixed-Lag Smoothing
*Estimate state at time $t-L$ using evidence up to time $t$.*

We keep a lag window of size $L$. For each step we compute:

$$
P(X_{t-L} | e_{1:t}) \propto \mathbf{f}_{t-L} \cdot \prod_{k=t-L+1}^{t} \mathbf{T}^\top \mathbf{O}_{e_k} \cdot \mathbf{1}
$$

In practice, the smoother stores intermediate forward messages and applies a backward correction using the future observations.

### Variable Definitions

| Symbol | Meaning |
|--------|---------|
| $X_t$ | Hidden state at day $t$ (Rain or Sun) |
| $e_t$ | Evidence at day $t$ (Umbrella or No umbrella) |
| $L$ | Lag window size (set by user) |
| $a$ | Transition persistence (slider) |
| $b$ | Sensor accuracy (slider) |
| $\mathbf{f}_t$ | Filtered belief vector at day $t$ |
| $\mathbf{O}_{e_t}$ | Diagonal matrix of $P(e_t|X_t)$ for the observed evidence |
| $\mathbf{T}$ | Transition matrix |

---

**Note:** All probabilities are normalised after each update.
"""
