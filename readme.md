# Braintorch

- release: 0.4


## Usage
```py
from braintorch.dataset import SignalDataset, SignalTestDataset

dataset = SignalDataset("dataset/train/train")

for signal, label in dataset:
    print(signal, label)
    break
```

For baseline correction, use the following code:
```py
from braintorch.utils import baseline_snip

baseline = baseline_snip(signal)
signal = signal - baseline
print(signal.shape)
```

For ICA, use the following code:
```py
from braintorch.utils import kurtosis_ica_method
clean_signal = kurtosis_ica_method(raw_signal)
```


## Example
```py
from braintorch.dataset import SignalDataset, SignalTestDataset
from braintorch.utils import baseline_snip, kurtosis_ica_method

dataset = SignalDataset(
    "train/train",
    baseline_snip,
    tans_segment_theory=2,
    acceptable_loss_sample=87
)
for segments, label in dataset:
    segments = kurtosis_ica_method(segments[:, :8])
    for channel in range(8):
        segment = segments[:, channel]
        print(segment.shape, label)
    break
```