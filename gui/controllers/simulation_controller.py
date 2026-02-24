"""
gui/controllers/simulation_controller.py
Acts as the intermediary between the GUI and the Smoothing Engine.
"""
import io
import sys
from contextlib import redirect_stdout
from typing import Tuple, List, Optional, Callable

from engine.smoother import FixedLagSmoother
from world.hmm import HiddenMarkovModel
from world.sensor import SensorModel
from world.transition import TransitionModel


class SimulationController:
    """Manages simulation state, engine interaction, and data preparation."""

    def __init__(self):
        self.smoother: Optional[FixedLagSmoother] = None
        self.days: List[int] = []
        self.forward_probs: List[float] = []
        self.smoothed_probs: List[float] = []

        # Store active parameters for logging
        self.current_f_alarm = 0.20

        # Callbacks to update the View
        self.on_log_update: Optional[Callable[[str], None]] = None
        self.on_insight_update: Optional[Callable[[str], None]] = None
        self.on_plot_update: Optional[Callable[[List, List, List, int], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None

    def initialize_world(self, lag: int, raw_s_acc: float, raw_t_per: float, raw_f_alarm: float) -> None:
        """Sanitizes inputs and initializes the Hidden Markov Model."""
        try:
            self._reset_state()

            # Guard constraints
            s_acc = max(0.001, min(0.999, raw_s_acc))
            t_per = max(0.001, min(0.999, raw_t_per))
            f_alarm = max(0.0, min(1.0, raw_f_alarm))
            self.current_f_alarm = f_alarm

            if abs(t_per - 0.5) < 1e-5:
                t_per = 0.5001  # Prevent singular matrices

            transition = TransitionModel([[t_per, 1 - t_per], [1 - t_per, t_per]])

            # Use the dynamic f_alarm instead of hardcoded 0.2
            sensor = SensorModel({
                True: [s_acc, f_alarm],
                False: [1 - s_acc, 1 - f_alarm]
            })

            hmm = HiddenMarkovModel(transition, sensor, [0.5, 0.5])
            self.smoother = FixedLagSmoother(hmm, lag=lag)

            self._broadcast_initial_state(t_per, s_acc, f_alarm, lag)
        except Exception as e:
            if self.on_error:
                self.on_error("Failed to initialize world engine.", e)

    def process_evidence(self, evidence: bool, s_acc: float) -> None:
        """Pushes new evidence to the engine and updates UI state."""
        if not self.smoother:
            if self.on_log_update:
                self.on_log_update("⚠️ **Error:** World not initialized. Apply configuration first.")
            return

        try:
            # Capture engine stdout
            captured = io.StringIO()
            with redirect_stdout(captured):
                smoothed_result = self.smoother.process_day(evidence)

            self._update_state(evidence, smoothed_result)
            self._broadcast_updates(evidence, s_acc, smoothed_result, captured.getvalue())

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error processing day {self.smoother.t}", e)

    def _reset_state(self) -> None:
        self.days.clear()
        self.forward_probs.clear()
        self.smoothed_probs.clear()

    def _update_state(self, evidence: bool, smoothed_result: Optional[Tuple]) -> None:
        self.days.append(self.smoother.t - 1)

        # 1. Safely grab and clamp the forward probability
        raw_f_prob = self.smoother.forward_filter.get_current_f()[0]
        f_prob = max(0.0, min(1.0, raw_f_prob))
        self.forward_probs.append(f_prob)

        # 2. Safely normalize and clamp the smoothed result
        if smoothed_result is not None:
            s_rain, s_sun = smoothed_result

            # Guard against absolute zero underflow to prevent division-by-zero crashes
            total = s_rain + s_sun
            if total <= 0:
                # If the smoothing matrix collapsed entirely, fallback to the filtered belief
                s_prob = f_prob
            else:
                # Strictly normalize
                s_prob = s_rain / total

            # Clamp between 0.0 and 1.0 to ensure the graph never breaks
            s_prob = max(0.0, min(1.0, s_prob))
            self.smoothed_probs.append(s_prob)

    def _broadcast_initial_state(self, t_per: float, s_acc: float, f_alarm: float, lag: int) -> None:
        if self.on_insight_update:
            self.on_insight_update("System ready. Click ☂️ or ☀️ to start.")
        if self.on_plot_update:
            self.on_plot_update(self.days, self.forward_probs, self.smoothed_probs, lag)
        if self.on_log_update:
            # Generate and send the detailed markdown introduction
            intro_text = self._generate_initial_explanation(t_per, s_acc, f_alarm, lag)
            self.on_log_update(intro_text)

    def _generate_initial_explanation(self, t_per: float, s_acc: float, f_alarm: float, lag: int) -> str:
        """Returns a detailed, beginner-friendly explanation of the initialized world."""
        return f"""
## SIMULATION ENGINE START – FIXED-LAG SMOOTHING

Welcome! This program simulates a **hidden state** (rain or sun) that you can't see directly.  
You only get indirect evidence: whether people carry an umbrella.

### World Parameters (what you set)
- **Weather persistence** = {t_per * 100:.1f}%  
  *If today is rainy, the chance tomorrow is also rainy is {t_per * 100:.1f}%.* *If today is sunny, the chance tomorrow is also sunny is {t_per * 100:.1f}%.*

- **Sensor accuracy** = {s_acc * 100:.1f}%  
  *When it rains, you see an umbrella with {s_acc * 100:.1f}% probability.* - **False Alarm Rate** = {f_alarm * 100:.1f}%  
  *When it's sunny, you might still see an umbrella (false alarm) {f_alarm * 100:.1f}% of the time (Mathematically independent from sensor accuracy).*

- **Lag window** = {lag} day(s)  
  *The algorithm waits {lag} extra days before giving its final estimate for a past day.* *This uses future evidence to refine the past (smoothing).*

### How it works – in plain English
1. **Filtering** (current day)  
   `P(rain today | all umbrella observations so far)`  
   This is the real-time belief, updated each day.

2. **Smoothing** (past day)  
   `P(rain {lag} days ago | all observations up to today)`  
   After we see {lag} more days of evidence, we can correct our old estimate.

### What you'll see
- The **plot** on the right shows both filtered (blue) and smoothed (orange/red) probabilities.
- The **log** below prints the mathematical details of every step.
- The **insight** panel gives a short, intuitive takeaway.

Click the **📐 View Mathematical Details** button to see the exact equations used.

Let's begin! Add your first observation with the buttons above.
"""

    def _broadcast_updates(self, evidence: bool, s_acc: float, smoothed_result: Optional[Tuple],
                           log_output: str) -> None:

        # Use dynamic f_alarm for correct logging of the matrix
        o_diag = [s_acc, self.current_f_alarm] if evidence else [1 - s_acc, 1 - self.current_f_alarm]
        ev_str = "UMBRELLA" if evidence else "CLEAR SKIES"

        # FIX: Grab the day we just processed out of our synchronized list
        current_day = self.days[-1]

        if self.on_log_update:
            header = f"\n## Day {current_day} – Observation: {ev_str}\n- Likelihood matrix: diag({o_diag[0]:.2f}, {o_diag[1]:.2f})\n"
            self.on_log_update(header + log_output)

        if self.on_insight_update:
            self.on_insight_update(self._generate_insight(evidence, smoothed_result))

        if self.on_plot_update:
            self.on_plot_update(self.days, self.forward_probs, self.smoothed_probs, self.smoother.lag)

    def _generate_insight(self, evidence: bool, smoothed_result: Optional[Tuple]) -> str:
        ev_str = "an umbrella" if evidence else "NO umbrella"
        current_f_pct = self.forward_probs[-1] * 100
        if smoothed_result is None:
            return f"Saw {ev_str}. Current rain belief: {current_f_pct:.1f}%. (Buffer filling...)"

        smoothed_day = self.days[-1] - self.smoother.lag
        filtered_at_smoothed = self.forward_probs[smoothed_day - 1] * 100

        # Calculate the delta safely
        s_rain_pct = smoothed_result[0] * 100 if sum(smoothed_result) > 0 else (smoothed_result[0] / sum(
            smoothed_result)) * 100 if sum(smoothed_result) != 0 else 0

        delta = s_rain_pct - filtered_at_smoothed
        direction = "increased" if delta > 0 else "decreased"
        return (f"By seeing {ev_str} today, the AI looked back to Day {smoothed_day} "
                f"and {direction} its rain belief by {abs(delta):.1f}%.")
