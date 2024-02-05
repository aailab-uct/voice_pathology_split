"""
Split voiced dataset into train and test sets randomly.
"""
from pathlib import Path
import random
from tqdm import tqdm

rnd = random.Random(42)

path_to_dataset = Path("datasets", "spectrogram_voiced")
files = sorted(list(path_to_dataset.glob("*.*")))
test = rnd.sample(files, k=208)


dataset_path = Path("datasets", "patients_random_segments_datasets_voiced")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

for spectrogram_path in tqdm(files):
    if "nonhealthy" in str(spectrogram_path):
        if spectrogram_path in test:
            dest = dataset_path.joinpath("test", "nonhealthy")
        else:
            dest = dataset_path.joinpath("train", "nonhealthy")
    else:
        if spectrogram_path in test:
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
