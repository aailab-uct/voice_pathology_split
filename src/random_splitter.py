"""
Split SVD dataset into train and test sets randomly.
"""
from pathlib import Path
import random
from tqdm import tqdm

random.seed(42)

path_to_dataset = Path("datasets", "spectrogram")
files = list(path_to_dataset.glob("*.*"))
random.shuffle(files)
test = random.sample(files, k=1985)


dataset_path = Path("datasets", "patients_random_segments_datasets")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)

for spectrogram_path in tqdm(list(path_to_dataset.glob("*.*"))):
    if "unhealthy" in str(spectrogram_path):
        if spectrogram_path in test:
            dest = dataset_path.joinpath("test", "unhealthy")
        else:
            dest = dataset_path.joinpath("train", "unhealthy")
    else:
        if spectrogram_path in test:
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
