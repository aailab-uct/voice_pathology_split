"""
This script creates and enumerates all the splited datasets with their full paths and creates a checksum from this string.
"""
from pathlib import Path
import hashlib
import shutil
from tqdm import tqdm


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
    shutil.rmtree('datasets/'+dataset, ignore_errors=True)

import split_all

p = Path("datasets")

files = p.glob("*/*/*/*.png")

names = []
for file in tqdm(files):
    names.append(file.as_posix())

print("SHA256 hash of list of files is:")
print(hashlib.sha256(str(sorted(names)).encode('utf-8')).hexdigest())

