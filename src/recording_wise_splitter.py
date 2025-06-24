"""
Use this script to split the SVD into train and test sets.
All spectrograms from each recording should be in either train or test set.
"""
# pylint: disable=bad-str-strip-call
from pathlib import Path
import random
from tqdm import tqdm

TRAIN_TEST_RATIO = 1/9
rnd = random.Random(42)

# Path to the spectrograms
path_to_dataset = Path("..","spectrograms", "svd")
# Path to final datasets for YOLO training
dataset_path = Path("..", "datasets", "segmentation_leakage_recording_wise_split_svd")
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)
# Extracting patient ids from the files
recording_ids = [filename.name.split("_")[2] for filename in path_to_dataset.glob("*.png")]
# Random shuffling of a list of patient ids so that they are randomly distributed into the training and test sets
recording_ids = sorted(list(set(recording_ids)))
rnd.shuffle(recording_ids)
# Random distribution of ids in
test_files = rnd.sample(recording_ids, k=int(TRAIN_TEST_RATIO * len(recording_ids)) + 1)
# Loop through the files and distribute them based on their patient id
for file in tqdm(list(path_to_dataset.glob("*.png"))):
    if file.name.split("_")[2] in test_files:
        folder = "test"
    else:
        folder = "train"
    health_state = file.name.split("_")[1]
    src = file.read_bytes()
    dataset_path.joinpath(folder, health_state, file.name).write_bytes(src)