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
from braintorch.utils import process_segment, kurtosis_ica_method
clean_signal = process_segment(raw_signal, kurtosis_ica_method)
clean_signal = kurtosis_ica_method(clean_signal)
```