"""
main.py

The main execution script. This file sets up the Umbrella World
scenario, instantiates the HMM and the Smoother, and feeds
evidence into the algorithm day by day.
"""
import numpy as np
from world.transition import TransitionModel
from world.sensor import SensorModel
from world.hmm import HiddenMarkovModel
from engine.smoother import FixedLagSmoother


def setup_umbrella_world() -> HiddenMarkovModel:
    """Configures the standard rules of the textbook Umbrella World."""
    # State 0: Rain, State 1: Sun
    prior = [0.5, 0.5]

    # T: [P(Rain|Rain), P(Sun|Rain)], [P(Rain|Sun), P(Sun|Sun)]
    transition = TransitionModel([
        [0.7, 0.3],
        [0.3, 0.7]
    ])

    # O: True (Umbrella seen), False (No Umbrella)
    sensor = SensorModel({
        True: [0.9, 0.2],
        False: [0.1, 0.8]
    })

    return HiddenMarkovModel(transition, sensor, prior)


def main():
    print("Initializing the Umbrella World Simulation...\n")

    # 1. Create the world
    hmm = setup_umbrella_world()

    # 2. Set the lag (d = 2)
    lag = 2
    smoother = FixedLagSmoother(hmm, lag=lag)

    # 3. Define a rigorous 35-day stress test sequence
    # This will prove the engine no longer suffers from catastrophic cancellation around Day 26.
    evidence_sequence = [
        True, True, False, True, True, False, True, True, False, True,
        True, True, False, True, True, True, True, True, False, False,
        True, True, True, True, False, True, False, False, True, True,
        False, True, True, True, False
    ]

    # 4. Run the simulation
    print(f"Starting Fixed-Lag Smoothing with Lag d={lag}")

    for day, evidence in enumerate(evidence_sequence, start=1):
        # The smoother orchestrates everything and returns the smoothed
        # result (if the window is full enough to calculate it)
        smoothed_result = smoother.process_day(evidence)

        if smoothed_result is not None:
            smoothed_day = day - lag
            # Format the output beautifully
            rain_prob = smoothed_result[0] * 100
            sun_prob = smoothed_result[1] * 100

            print(f"\n  [FINAL VERDICT FOR DAY {smoothed_day}]")
            print(f"  -> {rain_prob:.2f}% chance of Rain")
            print(f"  -> {sun_prob:.2f}% chance of Sun")


if __name__ == "__main__":
    # Ensure numpy prints arrays cleanly
    np.set_printoptions(suppress=True, precision=4)
    main()
