import warnings

import mne
import numpy as np
from pybaselines import Baseline
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def baseline_snip(signal: np.ndarray):
    range_sample = range(len(signal))
    baseline_fitter = Baseline(x_data=range_sample)
    baseline, _ = baseline_fitter.snip(signal, max_half_window=30, decreasing=True, smooth_half_window=3)
    return baseline


def kurtosis(data: np.ndarray):
    return ((data - np.mean(data)) ** 4).mean() / ((data - np.mean(data)) ** 2).mean() ** 2


def process_segment(segments: np.ndarray, l_freq: int = 7, h_freq: int = 30, SFreq: int = 250):
    segments_list = []
    for channel in range(segments.shape[1]):
        segment = segments[:, channel].astype(np.float64)
        segment = mne.filter.filter_data(
            data=segment, sfreq=SFreq, l_freq=l_freq, h_freq=h_freq, picks=None, verbose=False
        )
        segments_list.append(segment)

    segments_array = np.vstack(segments_list)
    return segments_array.T


def kurtosis_ica_method(segments: np.ndarray, n_components: int = 8, kurtosis_cutout: float = 8.5):
    eeg = process_segment(segments)
    # decompose EEG and plot components
    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        random_state=42,
    )
    ica.fit(eeg)
    components = ica.transform(eeg)

    kurtosis_scores = []
    for component_index in range(components.shape[1]):
        component = components[:, component_index]
        kurtosis_score = kurtosis(component)
        kurtosis_scores.append(kurtosis_score)

    kurtosis_scores = np.array(kurtosis_scores)
    remove_candidate_indices = np.where(kurtosis_scores > kurtosis_cutout)
    for remove_candidate_index in remove_candidate_indices:
        components[:, remove_candidate_index] = 0

    # Reconstruct EEG without blinks
    restored = ica.inverse_transform(components)
    return restored


def is_outlier(segment: np.ndarray):
    condition1 = stats.kurtosis(segment) > 4 * np.std(segment)
    condition2 = (abs(segment - np.mean(segment)) > 125).any()

    return condition1 or condition2
