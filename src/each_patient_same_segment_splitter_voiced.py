"""
This script takes the same (fifth) segment from each patient in voiced and puts it into test set.
Other segments from the same patient are put into train set.
"""
from pathlib import Path
from tqdm import tqdm

path_to_dataset = Path("datasets", "spectrogram_voiced")

dataset_path = Path("datasets", "recordings_same_segment_both_datasets_voiced")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

for spectrogram_path in tqdm(list(path_to_dataset.glob("*.*"))):
    if "nonhealthy" in str(spectrogram_path):
        if "00005" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "nonhealthy")
        else:
            dest = dataset_path.joinpath("train", "nonhealthy")
    else:
        if "00005" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
