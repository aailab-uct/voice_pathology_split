"""
The correct way to split the dataset. Use this script to split the voiced into train and test sets.
All spectrograms from each patient should be in either train or test set.
"""
# pylint: disable=bad-str-strip-call
from pathlib import Path
import random
from tqdm import tqdm

rnd = random.Random(42)

path_to_dataset = Path("datasets", "spectrogram_voiced")

dataset_path = Path("datasets", "patients_wise_datasets_voiced")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

patients_ids = []
for spectrogram_path in path_to_dataset.glob("*.*"):
    patients_ids.append(str(spectrogram_path.name).lstrip("voice")[:3])
patients_ids = sorted(list(set(patients_ids)))
rnd.shuffle(patients_ids)
test = rnd.sample(patients_ids, 24)
remove_from_test_set = rnd.sample(test, 4)
remove_segments = [rnd.randint(0, 9, ) for _ in range(4)]
for spectrogram_path in tqdm(sorted(list(path_to_dataset.glob("*.*")))):
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

# # randomly delete 8 files to get 208 samples in test set
# test_set_paths = list(dataset_path.joinpath("test").glob("**/*"))
# samples_to_delete = rnd.sample(test_set_paths, 8)

## As the glob is not deterministic and we forgot about it,
## we will remove the files that we know we should remove
samples_to_delete = [
    dataset_path.joinpath("test", "nonhealthy", "voice001_nonhealthy_8000_00006.png"),
    dataset_path.joinpath("test", "nonhealthy", "voice069_nonhealthy_8000_00003.png"),
    dataset_path.joinpath("test", "nonhealthy", "voice163_nonhealthy_8000_00004.png"),
    dataset_path.joinpath("test", "nonhealthy", "voice188_nonhealthy_8000_00006.png"),
    dataset_path.joinpath("test", "nonhealthy", "voice188_nonhealthy_8000_00000.png"),
    dataset_path.joinpath("test", "nonhealthy", "voice076_nonhealthy_8000_00002.png"),
    dataset_path.joinpath("test", "healthy", "voice096_healthy_8000_00004.png"),
    dataset_path.joinpath("test", "healthy", "voice177_healthy_8000_00001.png"),
]

for to_delete in samples_to_delete:
    to_delete.unlink()


