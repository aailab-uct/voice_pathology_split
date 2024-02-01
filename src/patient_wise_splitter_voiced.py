"""
The correct way to split the dataset. Use this script to split the voiced into train and test sets.
All spectrograms from each patient should be in either train or test set.
"""
# pylint: disable=bad-str-strip-call
from pathlib import Path
import random
from random import shuffle, sample


destination_path_spectrogram = Path("datasets", "spectrogram_voiced")

dataset_path = Path("datasets", "patients_wise_datasets_voiced")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

patients_ids = []
for spectrogram_path in destination_path_spectrogram.glob("*.*"):
    patients_ids.append(str(spectrogram_path.name).lstrip("voice")[:3])
patients_ids = list(set(patients_ids))
shuffle(patients_ids)
test = sample(patients_ids, 24)
remove_from_test_set = sample(test, 4)
remove_segments = [random.randint(0, 9, ) for _ in range(4)]
for spectrogram_path in destination_path_spectrogram.glob("*.*"):
    if "nonhealthy" in str(spectrogram_path):
        if str(spectrogram_path.name).lstrip("voice")[:3] not in test:
            dest = dataset_path.joinpath("train", "nonhealthy")
        else:
            dest = dataset_path.joinpath("test", "nonhealthy")
    else:
        if str(spectrogram_path.name).lstrip("voice")[:3] not in test:
            dest = dataset_path.joinpath("train", "healthy")
        else:
            dest = dataset_path.joinpath("test", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)

# randomly delete 8 files to get 208 samples in test set
test_set_paths = list(dataset_path.joinpath("test").glob("**/*"))
samples_to_delete = sample(test_set_paths, 8)

for to_delete in samples_to_delete:
    to_delete.unlink()
