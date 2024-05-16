import os
import re
from typing import Callable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_outlier, kurtosis_ica_method

SIGNAL_OPERATION = Callable[[np.ndarray], np.ndarray]


class SignalTestDataset(Dataset):
    def __init__(
        self,
        test_data_path: str,
        baseline_correction: Optional[SIGNAL_OPERATION] = None,
        filter_function: Optional[SIGNAL_OPERATION] = None,
    ):
        self.test_data_path = test_data_path
        self.baseline_correction = baseline_correction
        self.filter_function = filter_function

        data_path = os.listdir(self.test_data_path)
        self.signals = []

        # Load all the signals.
        for path in tqdm(data_path):
            path = os.path.join(self.test_data_path, path)

            segment = np.load(path)
            # Apply baseline correction if provided.
            if baseline_correction != None:
                segment -= baseline_correction(segment)
            # Apply filter function if provided.
            if filter_function != None:
                segment = filter_function(segment)

            self.signals.append(segment)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx) -> np.ndarray:
        return self.signals[idx]


class SignalDataset(Dataset):
    def __init__(
        self,
        training_data_path: str,
        baseline_correction: Optional[SIGNAL_OPERATION] = None,
        filter_function: Optional[SIGNAL_OPERATION] = None,
        max_range: int = 1750,
        acceptable_loss_sample: int = 15,
        tans_segment_theory: int = 0,
        apply_ica: bool = False,
        n_channels: int = 8,
    ):
        self.training_data_path = training_data_path
        self.baseline_correction = baseline_correction
        self.filter_function = filter_function
        self.acceptable_loss_sample = acceptable_loss_sample
        self.max_range = max_range
        self.tans_segment_theory = tans_segment_theory
        self.apply_ica = apply_ica
        self.n_channels = n_channels

        # Get all the data paths.
        data_path = self.get_data(self.training_data_path)
        self.signals = []
        self.labels = []

        # Append all the signals and labels.
        for path in tqdm(data_path):
            signal_path, label_path, signal_times, label_times = path
            signal = np.load(signal_path)
            label = np.load(label_path)
            signal_times = np.load(signal_times)
            label_times = np.load(label_times)

            segments_and_labels = self.get_segments_and_labels(
                signal,
                label,
                signal_times,
                label_times,
                self.baseline_correction,
                self.filter_function,
            )
            for segments in segments_and_labels:
                segment, label = segments
                self.signals.append(segment)
                self.labels.append(label)

    def get_data(self, path: str) -> List[Tuple[str, str, str, str]]:
        files = os.listdir(path)
        subject_id_list = []

        for file in files:
            # Search for the subject id.
            result = re.search("(s.*p[0-9]+)_([0-9]+)", file)
            if result.group(1) not in subject_id_list:
                subject_id_list.append(result.group(1))

        pairs_data = []
        for subject in subject_id_list:
            # Assume that there are 8 blocks of data for each subject.
            for i in range(8):
                # Get the signal and label paths.
                signal_path = "_".join([subject, str(i).zfill(3), "data", "time", "series"]) + ".npy"
                label_path = "_".join([subject, str(i).zfill(3), "label", "time", "series"]) + ".npy"
                signal_times = "_".join([subject, str(i).zfill(3), "data", "time", "stamps"]) + ".npy"
                label_times = "_".join([subject, str(i).zfill(3), "label", "time", "stamps"]) + ".npy"

                signal_path = os.path.join(path, signal_path)
                label_path = os.path.join(path, label_path)
                signal_times_path = os.path.join(path, signal_times)
                label_times_path = os.path.join(path, label_times)

                # Check if the files exist.
                if (
                    not os.path.exists(signal_path)
                    or not os.path.exists(label_path)
                    or not os.path.exists(signal_times_path)
                    or not os.path.exists(label_times_path)
                ):
                    continue
                pairs_data.append((signal_path, label_path, signal_times_path, label_times_path))

        return pairs_data

    def get_segments_and_labels(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        signal_times: np.ndarray,
        label_times: np.ndarray,
        baseline: Optional[SIGNAL_OPERATION] = None,
        signal_filter: Optional[SIGNAL_OPERATION] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_signals = []

        for i, channel in enumerate(signals.T):
            if i == len(signals.T) - 1:
                all_signals.append(channel)
                continue
            # Apply baseline correction if provided.
            if baseline != None:
                channel -= baseline(channel)
            # Apply filter function if provided.
            if signal_filter != None:
                channel = signal_filter(channel)

            all_signals.append(channel)

        all_signals = np.array(all_signals).T
        slices = []

        for i, label_time in enumerate(label_times):
            # Skip the first few segments.
            # To avoid fuctuation in the signal.
            if i < self.tans_segment_theory:
                continue

            start_index = np.abs(label_time - signal_times).argmin()
            signal = all_signals[start_index : start_index + self.max_range]

            # Remove the signal if it is less than the max range.
            if signal.shape[0] < self.max_range:
                continue
            # Remove the signal if it has more than acceptable loss sample.
            if len(np.where(signal[:, -1] == 0)[0]) > self.acceptable_loss_sample:
                continue

            signal = signal[:, : self.n_channels]
            if self.apply_ica:
                signal = kurtosis_ica_method(signal)
                if is_outlier(signal):
                    continue

            slices.append(signal)

        return tuple(zip(slices, labels))

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.signals[idx], self.labels[idx]
