"""
This script takes the same (fifth) segment from each patient in SVD and puts it into test set.
Other segments from the same patient are put into train set.
"""
from pathlib import Path
from tqdm import tqdm

path_to_dataset = Path("datasets", "spectrogram")

dataset_path = Path("datasets", "recordings_same_segment_both_datasets")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)

for spectrogram_path in tqdm(list(path_to_dataset.glob("*.*"))):
    if "unhealthy" in str(spectrogram_path):
        if "00005" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "unhealthy")
        else:
            dest = dataset_path.joinpath("train", "unhealthy")
    else:
        if "00005" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
