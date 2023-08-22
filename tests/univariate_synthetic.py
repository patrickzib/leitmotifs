import numpy as np
import pandas as pd
from motiflets.plotting import *


def generate_rectangular_wave(num_steps, signal_start=0):
    signal_duration = 50

    return (signal_duration, signal_start,
            np.where(np.arange(signal_duration) < signal_duration / 2, 1, -1))


def generate_sine_wave(num_steps, sine_start=0):
    sine_frequency = 0.05
    sine_duration = 100
    sine_phase = np.pi / 2
    # sine_start = np.random.randint(0, num_steps //  2 - sine_duration)
    # Generate sine wave
    sine_time = np.arange(sine_duration)
    sine_wave = np.sin(
        2 * np.pi * sine_frequency * sine_time + sine_phase)
    return sine_duration, sine_start, sine_wave


def test_random_walk():
    # Parameters
    num_steps = 2000
    random_walk_step_size = 0.1

    # Generate random walk
    time_series = np.random.normal(random_walk_step_size, size=num_steps)

    # Randomly choose the position to insert sine wave
    sine_start = [200, 400, 600]
    rect_start = [1000, 1200, 1400, 1600, 1800]
    for i in range(3):
        # Implant sine wave into random walk
        duration, start, sine = generate_sine_wave(num_steps, sine_start[i])
        noise = np.random.normal(0.1, size=duration) * 0.1
        time_series[start:start + duration] = sine + noise

    for i in range(5):
        duration, start, rect_wave = generate_rectangular_wave(num_steps, rect_start[i])
        noise = np.random.normal(0.1, size=duration) * 0.1
        time_series[start:start + duration] = rect_wave + noise

    ds_name = "Random Walk"
    df = pd.DataFrame(time_series).T

    ml = Motiflets(ds_name=ds_name, series=df, slack=1)
    ml.plot_dataset()

    k_max = 7
    length_range = np.arange(40, 110, 1)
    ml.fit_motif_length(k_max, length_range, subsample=1)
