"""
Take one random segment from each VOICED recording spectrogram and put it into the test set.
Other segments from the same recording are put into the train set.
"""
from pathlib import Path
import random
from tqdm import tqdm

def sorted_directory_listing_with_pathlib_glob(path_object):
    """
    Return a sorted list of directory items.
    """
    items = path_object.glob('*.*')
    sorted_items = sorted(items, key=lambda item: item.name)
    return [item.name for item in sorted_items]


rnd = random.Random(42)
# Path to the spectrograms
path_to_dataset = Path("..", "spectrograms", "voiced")
# Path to final datasets for YOLO training
dataset_path = Path("..", "datasets", "segmentation_leakage_random_segment_of_each_recording_voiced")
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

# List of all file names sorted alphabetically
sorted_paths = sorted_directory_listing_with_pathlib_glob(path_to_dataset)
# Placeholder for checking whether this recording has been already handled in the loop
actual_patient = "" # pylint: disable=invalid-name
for spectrogram_path_str in tqdm(sorted_paths):
    # Creating the destination path for each spectrogram
    spectrogram_path = path_to_dataset.joinpath(spectrogram_path_str)
    # Checks if it's the segment of the recording handled in the previous iteration
    if spectrogram_path_str.lstrip("voice")[:3] != actual_patient:
        # If this iteration handles a new recording, a random segment is selected and added to the test set
        actual_patient = spectrogram_path_str.lstrip("voice")[:3]
        random_segment = rnd.randint(0, 8)
    # Based on the file name, the spectrogram is placed in the correct class (healthy / unhealthy)
    # and correct set (training / testing based on the randomly selected segment number)
    if "nonhealthy" in str(spectrogram_path):
        if f"{random_segment:05}" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "nonhealthy")
        else:
            dest = dataset_path.joinpath("train", "nonhealthy")
    else:
        if f"{random_segment:05}" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    # Copying the spectrogram to the selected folder
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
