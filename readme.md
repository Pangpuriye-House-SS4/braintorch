# Braintorch

- release: 1.0


## Usage
```py
from braintorch.dataset import SignalDataset, RawSignalDataset

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
from braintorch.vis import visualtize_signals

dataset = SignalDataset(
    "train/train",
    baseline_snip,
    tans_segment_theory=2,
    acceptable_loss_sample=87,
    apply_ica=True,
)
for segments, label in dataset:
    visualtize_signals(
        segments, 
        distance= 20, 
        figsize=(12, 26)
    )
    break
```