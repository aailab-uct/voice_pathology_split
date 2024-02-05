"""
This script enumerates all the created datasets with their full paths and creates a checksum from this string.
"""
from pathlib import Path
import hashlib
import shutil
from tqdm import tqdm


p = Path("datasets")

# first level is the dataset name, second level is the type of the dataset (train, test)
# and third level is the class (healthy, unhealthy)
files = p.glob("*/*/*/*.png")

names = []
for file in tqdm(files):
    names.append(file.as_posix())

names1 = set(names)

datasets_folders = [
    'patients_random_segments_datasets',
    'patients_random_segments_datasets_voiced',
    'patients_random_segment_both_datasets',
    'patients_random_segment_both_datasets_voiced',
    'patients_same_segment_both_datasets',
    'patients_same_segment_both_datasets_voiced',
    'patients_wise_datasets',
    'patients_wise_datasets_voiced',
]

for dataset in datasets_folders:
    shutil.rmtree('datasets/'+dataset)

import split_all

p = Path("datasets")

files = p.glob("*/*/*/*.png")

names = []
for file in tqdm(files):
    names.append(file.as_posix())

names2 = set(names)

print("The difference between the two sets is:")
print(names1 - names2)

print("SHA256 hash of all list of files is:")
print(hashlib.sha256(str(names).encode()).hexdigest())
