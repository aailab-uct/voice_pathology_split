"""
Take one random segment from each patient in SVD and put it into the test set.
Other segments from the same patient are put into the train set.
"""
from pathlib import Path
import random
from tqdm import tqdm

rnd = random.Random(42)

path_to_dataset = Path("datasets", "spectrogram_voiced")

dataset_path = Path("datasets", "patients_random_segment_both_datasets_voiced")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "nonhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "nonhealthy").mkdir(exist_ok=True, parents=True)

def sorted_directory_listing_with_pathlib_glob(path_object):
    """
    Return a sorted list of directory items.
    """
    items = path_object.glob('*.*')
    sorted_items = sorted(items, key=lambda item: item.name)
    return [item.name for item in sorted_items]

sorted_paths = sorted_directory_listing_with_pathlib_glob(path_to_dataset)
actual_patient = "" # pylint: disable=invalid-name
for spectrogram_path_str in tqdm(sorted_paths):
    spectrogram_path = path_to_dataset.joinpath(spectrogram_path_str)
    if spectrogram_path_str.lstrip("voice")[:3] != actual_patient:
        actual_patient = spectrogram_path_str.lstrip("voice")[:3]
        random_segment = rnd.randint(0, 8)
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
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
