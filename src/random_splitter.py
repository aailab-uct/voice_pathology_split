"""
Split SVD dataset into train and test sets randomly.
"""
from pathlib import Path
import random
from tqdm import tqdm


TRAIN_TEST_RATIO = 1/9
rnd = random.Random(42)

path_to_dataset = Path("..", "spectrograms", "svd")
segments = list(path_to_dataset.glob("*.png*"))
test_files = rnd.sample(segments, k=int(TRAIN_TEST_RATIO * len(segments)) + 1)
train_files = set(segments) - set(test_files)

dataset_path = Path("..", "datasets", "segmentation_leakage_random_split_svd")
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)

for subset, folder in zip([train_files, test_files], ["train", "test"]):
    for spectrogram_path in tqdm(subset):
        health_state = spectrogram_path.name.split("_")[1]
        src = spectrogram_path.read_bytes()
        dataset_path.joinpath(folder, health_state, spectrogram_path.name).write_bytes(src)
