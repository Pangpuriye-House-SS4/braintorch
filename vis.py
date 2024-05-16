import matplotlib.pyplot as plt
import numpy as np


def visualtize_signals(signal: np.ndarray, distance: int = 20, figsize=(12, 26), lines=None):
    n_channels = signal.shape[1]
    plt.figure(figsize=figsize)
    plt.subplots(n_channels, 1, 1)

    for channel in range(8):
        segment = signal[:, channel]
        if lines == None:
            lines = range(len(segment))

        plt.plot(lines, segment + (channel * distance), label=channel)

    plt.legend()
    plt.show()
