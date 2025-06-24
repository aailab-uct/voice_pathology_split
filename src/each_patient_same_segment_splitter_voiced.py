"""
This script takes the same (fifth) segment from each recording in voiced and puts it into test set.
Other segments from the same recording are put into train set.
"""
from pathlib import Path
from tqdm import tqdm

# Path to the spectrograms
path_to_dataset = Path("..", "spectrograms", "voiced")
# Path to final datasets for YOLO training
dataset_path = Path("..", "datasets", "segmentation_leakage_same_segment_of_each_recording_voiced")
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

# Looping through all file names
for spectrogram_path in tqdm(list(path_to_dataset.glob("*.*"))):
    # Split based on the class (healthy / unhealthy) and segment number(5th segment -> test set)
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
    # Copying the spectrogram to the selected folder
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
