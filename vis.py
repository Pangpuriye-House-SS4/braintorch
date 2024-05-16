import matplotlib.pyplot as plt
import numpy as np


def visualtize_signals(signal: np.ndarray, distance: int = 20, figsize=(12, 26)):
    n_channels = signal.shape[1]
    plt.figure(figsize=figsize)
    plt.subplot(n_channels, 1, 1)

    for channel in range(8):
        segment = signal[:, channel]
        plt.plot(segment + (channel * distance), label=channel)

    plt.legend()
    plt.show()
