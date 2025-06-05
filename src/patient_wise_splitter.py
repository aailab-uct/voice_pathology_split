"""
The correct way to split the dataset. Use this script to split the SVD into train and test sets.
All spectrograms from each patient should be in either train or test set.
It uses maximum of one healthy and one unhealthy recording from each patient.
"""
# pylint: disable=bad-str-strip-call
from pathlib import Path
import random
from tqdm import tqdm

rnd = random.Random(42)

path_to_dataset = Path("datasets", "spectrogram")

dataset_path = Path("datasets", "patients_wise_datasets")
dataset_path.mkdir(exist_ok=True)
dataset_path.joinpath("train", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("train", "unhealthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "healthy").mkdir(exist_ok=True, parents=True)
dataset_path.joinpath("test", "unhealthy").mkdir(exist_ok=True, parents=True)

patients_ids = []
for spectrogram_path in path_to_dataset.glob("*.*"):
    patient_id = str(spectrogram_path.name).lstrip('svdadult')[:4]
    patient_rec_order = str(spectrogram_path.name).split('.')[0].split('_')[-2]
    if int(patient_rec_order.lstrip('order')) == 0:
        patients_ids.append(str(spectrogram_path.name).lstrip("svdadult")[:4])

patients_ids = sorted(list(set(patients_ids)))
rnd.shuffle(patients_ids)
test = rnd.sample(patients_ids, 184) # 90/10 split
for spectrogram_path in tqdm(sorted(list(path_to_dataset.glob("*.*")))):
    spectogram_order = str(spectrogram_path.name).split('.')[0].split('_')[-2]
    if int(spectogram_order.lstrip('order')) != 0:
        continue

    if "unhealthy" in str(spectrogram_path):
        if str(spectrogram_path.name).lstrip("svdadult")[:4] not in test:
            dest = dataset_path.joinpath("train", "unhealthy")
        else:
            dest = dataset_path.joinpath("test", "unhealthy")
    else:
        if str(spectrogram_path.name).lstrip("svdadult")[:4] not in test:
            dest = dataset_path.joinpath("train", "healthy")
        else:
            dest = dataset_path.joinpath("test", "healthy")
    src =spectrogram_path.read_bytes()
    dest.joinpath(spectrogram_path.name).write_bytes(src)
