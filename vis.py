import matplotlib.pyplot as plt
import numpy as np


def visualtize_signals(signal: np.ndarray, figsize=(12, 26), lines=None):
    n_channels = signal.shape[1]
    fig, axs = plt.subplots(n_channels, 1, figsize=figsize)

    for channel in range(8):
        segment = signal[:, channel]
        if lines == None:
            lines = range(len(segment))

        axs[channel].plot(lines, segment, label=channel)

    plt.show()
