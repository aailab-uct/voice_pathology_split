"""
Take one random segment from each patient in SVD and put it into the test set.
Other segments from the same patient are put into the train set.
"""
from pathlib import Path
import random

destination_path_spectrogram = Path("datasets", "spectrogram")

dataset_path = Path("datasets", "patients_random_segment_both_datasets")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)

def sorted_directory_listing_with_pathlib_glob(path_object):
    """
    Return a sorted list of directory items.
    """
    items = path_object.glob('*.*')
    sorted_items = sorted(items, key=lambda item: item.name)
    return [item.name for item in sorted_items]

sorted_paths = sorted_directory_listing_with_pathlib_glob(destination_path_spectrogram)
actual_patient = "" # pylint: disable=invalid-name
for spectrogram_path_str in sorted_paths:
    spectrogram_path = destination_path_spectrogram.joinpath(spectrogram_path_str)
    if spectrogram_path_str.lstrip("svdadult")[:4] != actual_patient:
        actual_patient = spectrogram_path_str.lstrip("svdadult")[:4]
        random_segment = random.randint(0, 8)
    if "unhealthy" in str(spectrogram_path):
        if f"{random_segment:05}" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "unhealthy")
        else:
            dest = dataset_path.joinpath("train", "unhealthy")
    else:
        if f"{random_segment:05}" in str(spectrogram_path):
            dest = dataset_path.joinpath("test", "healthy")
        else:
            dest = dataset_path.joinpath("train", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
