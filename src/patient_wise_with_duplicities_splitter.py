"""
Script for patient-wise splitting. Use this script to split the SVD into train and test sets.
All spectrograms from each recording should be in either train or test set.
It uses ALL recordings from each patient.
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
dataset_path = Path("..", "datasets", "patients_wise_split_svd_with_duplicities")
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)
# Extracting patient ids from the files
patients_ids = [filename.name.lstrip("svdadult")[:4] for filename in path_to_dataset.glob("*.png")]
# Random shuffling of a list of patient ids so that they are randomly distributed into the training and test sets
patients_ids = sorted(list(set(patients_ids)))
rnd.shuffle(patients_ids)
# Looping through the list of ids and filling the train/test folders
folder = "test"
# counting only the first healthy and pathological files for each patient id
total_file_count = len(list(path_to_dataset.glob("*.png")))
for id in tqdm(patients_ids):
    # Count the ratio of files in a training set to the total number of files
    test_set_count = len(list(dataset_path.joinpath("test").glob("**/*.png")))
    # If the count reaches over the train-test split ratio, saving recordings to the train subfolders
    if (test_set_count / total_file_count >= TRAIN_TEST_RATIO) and (folder == "test"):
        folder = "train"
    # Loop through each recording under the id and move it to the correct subfolder
    # Considering only the file names with order00 in their name as these indicate the first healthy and pathological
    # recordings for each patient -> ensure only single healthy or pathological recording for each patient id
    for segment in path_to_dataset.glob(f"svdadult{id}*.png"):
        health_state = segment.name.split("_")[1]
        src = segment.read_bytes()
        dataset_path.joinpath(folder, health_state, segment.name).write_bytes(src)

